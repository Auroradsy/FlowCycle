# FlowCycle


**A CycleGAN rewired so that a single invertible flow *is* the bridge between modalities.**

Cross-modal translation between paired ADNI T1 and FA brain slices, where the two
modalities share one representation space joined by one bijection — not two
independent translators that happen to be trained together.

---

## The idea

Plain CycleGAN carries two generators, `G_A→B` and `G_B→A`. Each is a complete,
self-contained translator; nothing is shared between them, and neither has a
representation you can inspect, interpolate, or reuse.

CycleFlow **splits** those two generators instead of augmenting them. Each
`ResnetGenerator` is cut at its bottleneck into a front half and a back half:

```
G_A→B  =  head + down   ‖   res + up + tail
             E_A                  D_B

G_B→A  =  head + down   ‖   res + up + tail
             E_B                  D_A
```

The four halves are then re-joined through one convolutional normalizing flow `f`:

```
                    T1 ──E_A──►  z  ──D_A──► T1          (self)
                                 │ ▲
                                 f f⁻¹
                                 ▼ │
                    FA ──E_B──►  u  ──D_B──► FA          (self)

        T1 → FA :   T1 ─E_A→ z ─f→ u ─D_B→ FA            (cross)
        FA → T1 :   FA ─E_B→ u ─f⁻¹→ z ─D_A→ T1          (cross)
```

`z` is A's representation, `u` is B's; both are 256×28×28 feature maps.

Three properties this buys, each of them checkable rather than asserted:

1. **Parameter-neutral host.** `E_A + D_B` is exactly `G_A→B` and `E_B + D_A` is
   exactly `G_B→A`, so the backbone costs precisely what plain CycleGAN costs.
   The flow adds 1.84 M (+11.8 %).
2. **`f` is the exact identity at initialisation.** `SpatialActNorm` starts at
   `log_scale = bias = 0` and `SpatialCoupling`'s last conv is zero-init, so a
   warm-started model reproduces its CycleGAN host *bit-for-bit* on both cross
   directions. Every gain is therefore attributable to the rewiring rather than
   to a different training run. Verify with `--check_init` (below).
3. **No bypass.** `D_B` only ever sees `E_B(FA)` or `f(E_A(T1))`; there is no
   spatial path around the flow, so swapping in another subject's code must
   destroy the output by construction. It does — see the probe results below.

And one thing the rewiring gives away for free: because `f` is a *sequence* of
invertible blocks, its own intermediate states `z → b₁ → … → b_L = u` decode into
a **native morph** from one modality to the other. These are the flow's own block
outputs, not an interpolation between two codes — an interpolant would be
available to any autoencoder.

## Parameters

| | plain CycleGAN (×2, bidirectional) | CycleFlow |
|---|---|---|
| inference | 31.30 M | **17.49 M** (−44 %) |
| training | 42.34 M | **23.01 M** (base / latcyc) · 25.77 M (morph, +`D_mix`) |

One CycleGAN generator maps a single direction, so a bidirectional system is
counted as two. CycleFlow's bridge is **one** 1.84 M flow traversed forwards and
backwards, and `f⁻¹(f(z)) = z` holds exactly rather than being penalised into place.

---

## Training

Three stages, each with a different set of frozen parameters.

| | trains | frozen | losses |
|---|---|---|---|
| **S1** | `E_A E_B D_A D_B` | `f` (= identity) | `L_self`, `L_gan` |
| **S2** | `f` | `E`, `D` | `L_cross` |
| **S3** | `f`, `D_A D_B` | `E` | `L_cross`, `L_gan`, `L_cyc`, `L_latcyc`, `L_path` |

**S1 deliberately does not train the cross paths.** If it did, the two feature
spaces would align on their own and the flow would have nothing left to do —
`flow_work → 0`, the identity collapse we measured at 0.34 in an earlier run.
Keeping the two autoencoders independent guarantees `f` has real work.

**`E` stays frozen in S3** because the representation is the object of study.
Letting the encoders drift while the decoders adapt would turn any morph result
into a statement about the decoders instead.

### The two non-obvious losses

`L_latcyc` — the flow's output must be a **fixed point** of the target
autoencoder, i.e. decodable:

```
‖E_B(D_B(f z)) − f z‖₁ / ‖f z‖₁   +   ‖E_A(D_A(f⁻¹u)) − f⁻¹u‖₁ / ‖f⁻¹u‖₁
```

The normalisation makes it scale-free, so the flow cannot satisfy it by shrinking
its output. Crucially it constrains *decodability*, not the flow's
**distribution** — the latter is what a NLL term does, at the cost of collapsing
`flow_work`.

`L_path` — the intermediate states must decode to something real and must move
smoothly, with `t ~ U{1…L}` sampled once per step:

```
realism      (D_mix(D_B(s_{t−1})) − 1)²  +  (D_mix(D_B(s_t)) − 1)²      w = 0.5
smoothness   ‖D_B(s_t) − D_B(s_{t−1})‖₁                                 w = 1.0
```

`D_mix` is a third discriminator whose "real" set is **T1 ∪ FA pooled**, so a
half-way frame is a legal answer. Using `D_FA` here instead flattens the handover
into a step. **No ground-truth intermediate frames are needed** — that is the
point.

### Variants

| variant | `L_latcyc` | `L_path` | `D_mix` |
|---|---|---|---|
| `base` | — | — | no |
| `latcyc` | ✓ | — | no |
| `morph` | ✓ | ✓ | yes |

---

## Results

T1↔FA, MNI-registered axial slices z ∈ [40, 49], 170 train / 43 test subjects
(1700 / 430 slices), split by subject.

| | self T1 | self FA | T1→FA | FA→T1 | `flow_work` | `latent_gap` |
|---|---|---|---|---|---|---|
| plain CycleGAN (host) | — | — | 0.680 | 0.813 | — | — |
| `base` | 0.9995 | 0.9993 | **0.7626** | **0.8421** | 1.897 | 2.216 |
| `latcyc` | 0.9983 | 0.9979 | 0.7497 | 0.8195 | 1.155 | 0.841 |
| `morph` | 0.9984 | 0.9980 | 0.7545 | 0.8266 | 1.184 | 0.816 |

The rewiring alone (`base`) beats the host it was cut from by **+0.083 T1→FA**
and **+0.030 FA→T1**, and adds a self-reconstruction path CycleGAN does not have
at all (SSIM 0.999).

**The morph terms are non-substitutable.** Per-frame `mean|Δ|` at the path
endpoint, decoded by both decoders:

| variant | `D_A` (T1 view) | `D_B` (FA view) | |
|---|---|---|---|
| `base` | **0.151** | 0.059 | change appears in the *wrong* view — the T1 view tears |
| `latcyc` | 0.058 | 0.043 | no tearing, but nothing happens |
| `morph` | 0.057 | **0.156** | usable |

`L_latcyc` stops the damage (0.151 → 0.058); `L_path` creates the progression
(0.043 → 0.156). Neither term does the other's job.

### Read SSIM on this data with care

MNI-registered slices are so well aligned that raw SSIM is nearly
uninterpretable. `eval.py` reports the two floors a number has to
clear before it means anything:

| | FA target | T1 target |
|---|---|---|
| `morph`, correct code | 0.7545 | 0.8266 |
| `morph`, **another subject's** code | 0.6589 | 0.6903 |
| inter-subject floor (a different real brain) | 0.6456 | 0.6790 |
| **template floor** (the test-set mean image — ignores the input entirely) | **0.7134** | **0.7562** |

A cross-modal SSIM that does not clear the template floor is not measuring
subject-specific fidelity, however it ranks against other methods. The u-shuffle
drop of **0.096 / 0.136** to essentially the inter-subject floor is the evidence
that the shared latent is load-bearing rather than decorative.

> Note on the probe: it must permute **across subjects**. The test loader is
> unshuffled and each subject contributes 10 consecutive axial slices, so a
> roll-by-one hands the decoder the *same subject's neighbouring slice* and
> proves nothing. `cross_subject_perm()` asserts no element keeps its own subject.

> Note on the axial band: `python -m data.build_cache --report` measures how much
> of the variance is slice position rather than subject identity. The middle-10
> band cuts that ratio from 1.73 to 1.29 — a real reduction, but position still
> moves the image more than identity does. Read cross-subject numbers on this
> data with that in mind.

---

## Layout

```
CycleFlow/
├── train.py                the three-stage trainer
├── train_host.py           plain CycleGAN — produces the warm-start checkpoint
├── eval.py                 cross-subject u-shuffle probe + both SSIM floors
│
├── configs/                hyperparameters and launch scripts
│   ├── base.yaml  latcyc.yaml  morph.yaml  morph_bi.yaml
│   ├── _env.sh             sourced by every script: repo root, conda env, GPU
│   ├── run_base.sh  run_latcyc.sh  run_morph.sh  run_morph_bi.sh  run_host.sh
│   ├── check_init.sh       the attribution check — run this first
│   ├── run_all.sh          all three variants, then the figures
│   ├── run_s3_ablation.sh  shared S1+S2, S3 is the only difference
│   ├── run_path_search.sh  three ways out of the D_mix problem
│   └── make_figures.sh     probe + every plot
│
├── data/
│   ├── paired_dataset.py   dataset, subject-level split, axial band
│   ├── build_cache.py      NIfTI -> data/cache/paired_112.pt;  --report
│   ├── preprocess/         raw ADNI -> MNI 2mm (needs FSL)
│   │   ├── register_t1_mni.py    flirt + fnirt, T1 -> MNI152 2mm
│   │   ├── resample_dti_2mm.py   DTI scalars, MNI 1mm -> 2mm grid
│   │   ├── build_labels.py       -> labels.csv
│   │   └── qc_overlays.py        FA-edge-on-T1 overlays to eyeball alignment
│   ├── cache/              gitignored — 554 MB of slices
│   └── labels.csv          gitignored — ADNI subject IDs + diagnoses
│
├── model/
│   ├── backbone.py         ResnetGenerator, PatchDiscriminator, init_weights
│   ├── flow.py             SpatialActNorm, SpatialCoupling, SpatialFlow
│   └── mmclast.py          Encoder, Decoder, MMCLASTcg, make_discriminators
│
├── utils/
│   ├── image.py            to_pm1 / to_01 / ssim — the shared conventions
│   ├── pool.py             CycleGAN's ImagePool
│   ├── config.py           YAML under argparse, with the precedence rules
│   ├── plot_morph.py       figs 30 (a2b), 31, 33 (b2a)
│   └── plot_ablation.py    figs 32, 34, 35  (--direction, --out)
│
└── exps/                   ALL run output.  gitignored in full — nothing here
    │                       ships with the repo; regenerate it from configs/.
    │                       Relocate the whole tree with $MMCLAST_EXPS.
    ├── checkpoints/<tag>/  weights, final_eval.txt, probe_eval.txt
    │   └── host/           the CycleGAN checkpoint train.py splits
    ├── logs/<tag>/         train_log.csv (per epoch) + train.log (stdout)
    └── snapshot_results/   the figures
```

**Figure naming.** Every direction-dependent figure carries its direction in the
filename: `..._a2b_...` is T1 → FA (walk `f` forward), `..._b2a_...` is FA → T1
(walk `f` backward). Fig 30/33 are the per-run morphs, 31 is self-vs-cross, and
32/34/35 are ablations across several runs.

> The figures are **not** committed. A fresh clone therefore has no results to
> look at and, because the ADNI cache and the checkpoints are also gitignored,
> no way to regenerate them without the data. If you want the figures to travel
> with the repo, drop `exps/` from `.gitignore` and `git add -f
> exps/snapshot_results/` — 4 MB for the one thing a reader can understand
> without running anything.

`train_host.py` is here because it is the *host initialisation*, not because it
is a baseline for comparison: `check_init.sh` needs its checkpoint to show that
CycleFlow starts out identical to it.

## Configs

Each variant is one self-contained YAML. They deliberately repeat the shared
values rather than inheriting from a common base — a config you can read in one
screen is worth more than one you have to resolve across three files.

Precedence is **argparse defaults < `--config` file < flags you type**:

```bash
python train.py --config configs/morph.yaml                 # the file
python train.py --config configs/morph.yaml --w_latcyc 5    # 5 wins
```

An unknown key in a config is a hard error rather than a silent no-op — a typo'd
hyperparameter that does nothing is the kind of bug that only ever shows up as a
run which mysteriously fails to reproduce.

The variant presets in `train.py` are applied *after* the merge, so `w_latcyc`,
`w_path_gan` and `w_path_smooth` are pinned by `variant` regardless of the file.
A variant is a definition, not a default.

## Data

The dataset expects MNI-registered 2 mm volumes, one pair per subject:

```
$ADNI_T1_DIR/<subject>_T1_mni2.nii.gz     (91, 109, 91)
$ADNI_FA_DIR/<subject>_FA_mni2.nii.gz     (91, 109, 91)
$ADNI_T1_DIR/manifest.csv                 columns: subject, status
data/labels.csv                           columns: subject, label_2..label_6
```

`data/preprocess/` takes raw ADNI to that state (FSL flirt/fnirt; see each
script's docstring). Slices are then rotated to visual axial orientation,
centre-padded to 112×112, and per-slice normalised to [0, 1] clipped at the 99th
percentile. Training is unsupervised; labels only determine the split.

Override any path by environment variable: `ADNI_ROOT`, `ADNI_T1_DIR`,
`ADNI_FA_DIR`, `ADNI_CACHE`, `ADNI_LABELS`.

**ADNI data is not redistributable.** Neither the imaging cache nor `labels.csv`
(subject IDs and diagnoses) is included here, and both are gitignored. Obtain
access at <https://adni.loni.usc.edu/>.

## Running

```bash
pip install -r requirements.txt

# 0. build the slice cache, and see what the axial band actually buys you
python -m data.build_cache
python -m data.build_cache --report

# 1. the host, if you do not already have a CycleGAN checkpoint
bash configs/run_host.sh

# 2. confirm the rewiring is a no-op at initialisation.
#    Must print the host's own scores at flow_work = 0.
bash configs/check_init.sh

# 3. train.  Sequential on one GPU, or PARALLEL=1 for one variant per GPU.
bash configs/run_all.sh
#    or individually:
CUDA_VISIBLE_DEVICES=1 bash configs/run_morph.sh

# 4. probe + figures (run_all.sh already does this)
bash configs/make_figures.sh
```

Roughly 12 h per variant on a single A6000 at the default schedule
(S1 120 ep / S2 80 / S3 120, batch 8). Every script tees to `exps/logs/<tag>/train.log`.

### Knobs worth knowing

| key | default | |
|---|---|---|
| `variant` | `morph` | `base` / `latcyc` / `morph`; pins the three weights below |
| `warm` | `exps/checkpoints/host/last.pth` | CycleGAN checkpoint to split; `""` trains from scratch |
| `z_lo` / `z_hi` | 40 / 49 | axial band; see `build_cache --report` |
| `n_flow` | 4 | flow blocks — also the number of morph frames |
| `pre_relu` | 1 | tap the bottleneck *before* `down`'s last ReLU. Leave on: otherwise `feat ≥ 0`, and a non-negative code cannot host a signed flow output |
| `w_latcyc` | 2.0 | 0 in `base` |
| `w_path_gan` / `w_path_smooth` | 0.5 / 1.0 | 0 outside `morph` |
