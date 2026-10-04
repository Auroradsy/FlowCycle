# FlowCycle: Rethinking Cycle Consistency via Exact Bijective Latent Flows

Official PyTorch implementation of **FlowCycle**, a bidirectional image-translation
model in which a single invertible latent flow is the bridge between two modalities.

<p align="center">
  <img src="assets/framework.png" width="90%">
</p>

Encoders `E_A`, `E_B` map paired images to latents. A discrete normalizing flow `f`
(spatial affine coupling + ActNorm) is an exact, solver-free bijection between the
two latent spaces: `A → B` runs `f`, `B → A` runs `f⁻¹`, and decoders `D_B`, `D_A`
map the transported codes back to images. The intermediate states of the flow blocks
give an intrinsic transition trajectory between the modalities.

## Contents

- [Installation](#installation)
- [Data](#data)
- [Training](#training)
- [Evaluation](#evaluation)
- [Repository layout](#repository-layout)

## Installation

Python 3.11 and a CUDA build of PyTorch 2.5.1.

```bash
git clone https://github.com/Auroradsy/CycleFlow.git
cd CycleFlow

conda create -n flowcycle python=3.11 -y
conda activate flowcycle
pip install -r requirements.txt
```

All outputs (checkpoints, logs) are written under `$FLOWCYCLE_EXPS`, and datasets are
read from the paths below. Set them once per shell:

```bash
export FLOWCYCLE_EXPS=/path/to/exps        # run outputs
export DATA=/path/to/datasets              # used in the commands below
```

Checkpoints of a run land in `$FLOWCYCLE_EXPS/<dataset>/checkpoints/<tag>/`
(`stage1.pth`, `stage2.pth`, `stage3.pth`, `model.pth`).

## Data

FlowCycle trains on paired data. Two loaders are provided: `--data adni` (cached
T1/FA slices) and `--data folder` (image folders).

**Folder layout** (`--data folder`, with `pair: true`): the two members of a pair
share a file name.

```
<data_root>/
├── trainA/  trainB/      training pairs
└── testA/   testB/       held-out pairs
```

### Synthetic MNIST MRI/PET

Rendered from MNIST; `A` is the grayscale digit, `B` the PET-style rendering.

```bash
python -m data.make_mnist_petct --paired_train \
    --mnist_raw $DATA/MNIST/raw --out $DATA/mnist_petct_paired
```

### Cityscapes

Download the paired `cityscapes` set released with
[pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/datasets.md).
Each image is `[photo | label map]` side by side: split it into the left half (`A`,
photo) and the right half (`B`, labels) and store them in the folder layout above,
using pix2pix's `train` as `train*` and `val` as `test*`.

### ADNI T1 / FA

ADNI is not redistributable; request access at <https://adni.loni.usc.edu/>.
The loader expects MNI-registered 2 mm volumes, one pair per subject:

```
$ADNI_T1_DIR/<subject>_T1_mni2.nii.gz
$ADNI_FA_DIR/<subject>_FA_mni2.nii.gz
$ADNI_T1_DIR/manifest.csv          columns: subject, status
$ADNI_LABELS                       columns: subject, label_2..label_6
```

`data/preprocess/` takes raw ADNI to that state (requires FSL):

```bash
python data/preprocess/register_t1_mni.py     # T1 -> MNI152 2 mm (flirt + fnirt)
python data/preprocess/resample_dti_2mm.py    # FA -> the same 2 mm grid
python data/preprocess/build_labels.py        # -> labels.csv
```

Then build the slice cache (10 axial slices per subject, 112×112, subject-level
80/20 split):

```bash
export ADNI_CACHE=$DATA/ADNI/paired_112.pt
export ADNI_LABELS=$DATA/ADNI/labels.csv
python -m data.build_cache
```

## Training

Training has two parts: a CycleGAN **host** whose generators initialise the encoders
and decoders, then the three FlowCycle stages.

| stage | trains | objective |
|---|---|---|
| 1 | encoders, decoders | reconstruction + adversarial |
| 2 | flow | cross-modal transport |
| 3 | flow, decoders | transport + cycle (+ latent and trajectory regularization) |

Each stage stops early on validation loss and restores its best checkpoint.
Hyperparameters live in one YAML per run (`configs/`); any key can be overridden on
the command line, e.g. `--batch 16`.

Two models are reported in the paper:

- **FlowCycle w/o TR** — `variant: base`.
- **FlowCycle-TR** — adds the latent consistency and trajectory smoothness terms in
  stage 3 (`--variant morph --w_latcyc 2.0 --w_path_gan 0 --w_path_smooth 1.0`). It
  resumes the stage-2 checkpoint of the base run, so stage 3 is the only difference.

### ADNI

```bash
# 1. host
HOST_TAG=host python train_host.py

# 2. FlowCycle w/o TR
python train.py --config configs/base.yaml --tag adni_base

# 3. FlowCycle-TR, from the same stage 2
python train.py --config configs/morph.yaml --tag adni_tr \
    --resume_stage 2 --resume_from $FLOWCYCLE_EXPS/adni/checkpoints/adni_base/stage2.pth \
    --w_path_gan 0 --w_path_smooth 1.0
```

### Synthetic MNIST MRI/PET

```bash
R=$DATA/mnist_petct_paired
HOST=$FLOWCYCLE_EXPS/mnist/checkpoints/mnist_host/last.pth

# 1. host
HOST_TAG=mnist_host python train_host.py --data folder --data_root $R \
    --img_ch 3 --load_size 72 --crop_size 64 --no_flip \
    --epochs 100 --decay_start 50 --batch 64

# 2. FlowCycle w/o TR
python train.py --config configs/mnist_p_base.yaml --data_root $R --warm $HOST

# 3. FlowCycle-TR
python train.py --config configs/mnist_p_morph.yaml --data_root $R --warm $HOST \
    --tag mnist_p_tr \
    --resume_stage 2 --resume_from $FLOWCYCLE_EXPS/mnist/checkpoints/mnist_p_base/stage2.pth \
    --w_gan 0.1 --w_path_gan 0 --w_path_smooth 0.3
```

### Cityscapes

```bash
export CYCLEFLOW_DATASET=cityscapes
R=$DATA/cityscapes_paired
HOST=$FLOWCYCLE_EXPS/cityscapes/checkpoints/cityscapes_host/last.pth

# 1. host
HOST_TAG=cityscapes_host python train_host.py --data folder --data_root $R \
    --img_ch 3 --load_size 286 --crop_size 256 --n_blocks 9 \
    --epochs 160 --decay_start 80 --batch 8

# 2. FlowCycle w/o TR
python train.py --config configs/p2p_base.yaml --data_root $R --warm $HOST \
    --tag cityscapes_base

# 3. FlowCycle-TR
python train.py --config configs/p2p_base.yaml --data_root $R --warm $HOST \
    --tag cityscapes_tr \
    --resume_stage 2 --resume_from $FLOWCYCLE_EXPS/cityscapes/checkpoints/cityscapes_base/stage2.pth \
    --variant morph --w_latcyc 2.0 --w_path_gan 0 --w_path_smooth 1.0
```

## Evaluation

All scripts score the held-out split against the paired targets (SSIM / PSNR) and
read `model.pth` of the given tags.

```bash
# ADNI: reconstruction and translation, both directions
python -m utils.eval_adni_recon --tags adni_base adni_tr

# ADNI: latent-swap probe with inter-subject and template SSIM floors
python eval.py --tag adni_tr

# MNIST MRI/PET
python -m utils.eval_mnist --root $DATA/mnist_petct_paired --tags mnist_p_base mnist_p_tr
```

`utils/fid.py` computes FID between two image sets.

## Repository layout

```
├── train.py              three-stage FlowCycle trainer
├── train_host.py         CycleGAN host (initialises encoders / decoders)
├── eval.py               latent-swap probe and SSIM floors (ADNI)
├── server_paths.py       output and data path resolution
├── configs/              one YAML per run, plus launch scripts
├── data/                 datasets, MNIST MRI/PET rendering, ADNI preprocessing
├── model/
│   ├── backbone.py       ResNet generator, PatchGAN discriminator
│   ├── flow.py           SpatialActNorm, SpatialCoupling, SpatialFlow
│   └── flowcycle.py      Encoder, Decoder, FlowCycle
└── utils/                config loading, image metrics, evaluation
```

## Acknowledgements

The host network and the folder data layout follow
[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
