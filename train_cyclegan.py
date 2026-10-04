#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a CycleGAN baseline for T1<->FA cross-modal translation (ADNI pilot).

Standard CycleGAN (Zhu et al. 2017):
  - LSGAN adversarial loss (MSE)
  - cycle-consistency L1, lambda_cycle=10
  - identity L1, lambda_identity=5
  - Adam lr=2e-4, betas=(0.5, 0.999)
  - image buffer (pool of 50) for the discriminators
  - linear LR decay over the second half of training

Data is treated as UNPAIRED for training: each step samples a T1 batch and an
independently-shuffled FA batch. The pairing is only used at eval time.

Run:
  CUDA_VISIBLE_DEVICES=1 python train.py
"""
import os
import sys
import csv
import time
import random
import itertools
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data.paired_dataset import (
    build_cache, PairedADNISliceDataset, subject_level_split,
)
from model import ResnetGenerator, PatchDiscriminator, init_weights

# The CycleGAN weights are what train.py splits into E/D, so they live beside the
# FlowCycle runs.  CYCLEGAN_TAG picks the sub-directory (default: the mid-10 CycleGAN).
from server_paths import experiment_root, checkpoint_root
EXPS = experiment_root()
RESULTS = os.path.join(EXPS, "checkpoints", os.environ.get("CYCLEGAN_TAG", "cyclegan"))
LOGDIR = os.path.join(EXPS, "logs", os.environ.get("CYCLEGAN_TAG", "cyclegan"))


# ---------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ImagePool:
    """Buffer of previously generated images (CycleGAN trick, size 50)."""

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        out = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(img)
                out.append(img)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.pool_size - 1)
                    tmp = self.images[idx].clone()
                    self.images[idx] = img
                    out.append(tmp)
                else:
                    out.append(img)
        return torch.cat(out, dim=0)


def to_minus1_1(x):
    """[0,1] -> [-1,1]."""
    return x * 2.0 - 1.0


def to_01(x):
    """[-1,1] -> [0,1] clamped."""
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


class _Swapped(torch.utils.data.Dataset):
    """Serve (b, a, y) from a dataset that yields (a, b, y)."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        a, b, y = self.ds[i]
        return b, a, y


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--decay_start", type=int, default=80,
                    help="epoch at which linear LR decay to 0 begins")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ngf", type=int, default=64)
    ap.add_argument("--ndf", type=int, default=64)
    ap.add_argument("--n_blocks", type=int, default=6)
    ap.add_argument("--lambda_cycle", type=float, default=10.0)
    ap.add_argument("--lambda_id", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save_every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    # --- unpaired image folders (CycleGAN layout) -------------------------
    # `adni` is the original path and is untouched.  `folder` swaps in
    # <root>/{trainA,trainB,testA,testB}; there is no ground-truth pairing, so
    # the final SSIM/PSNR block is replaced by FID.
    ap.add_argument("--data", default="adni", choices=["adni", "folder"])
    ap.add_argument("--data_root", default="")
    ap.add_argument("--img_ch", type=int, default=1)
    ap.add_argument("--load_size", type=int, default=286)
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--dp", action="store_true",
                    help="split each batch across all visible GPUs with "
                         "nn.DataParallel.  Worth it only if the per-GPU batch "
                         "stays large enough to amortise the per-step replicate/"
                         "scatter/gather; raise --batch alongside it.")
    ap.add_argument("--swap_domains", action="store_true",
                    help="exchange the two domains, so the generator stored as "
                         "G_T1toFA is trained natively in the B->A (FA->T1) "
                         "direction; checkpoint keys keep their names")
    ap.add_argument("--no_flip", action="store_true",
                    help="disable h-flip augmentation; required for chiral "
                         "content such as digits, where a mirrored sample is "
                         "not a valid member of the domain")
    args = ap.parse_args()
    if args.data == "folder" and not args.data_root:
        ap.error("--data folder requires --data_root")

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  args={vars(args)}", flush=True)

    # --- data (train split only; treat as unpaired) ---
    if args.data == "adni":
        build_cache()
        tr_idx, te_idx = subject_level_split(seed=42, test_frac=0.20,
                                             label_scheme="label_4")
        tr_ds = PairedADNISliceDataset(tr_idx, label_scheme="label_4")
        te_ds = PairedADNISliceDataset(te_idx, label_scheme="label_4")
    else:
        from data.unpaired_dataset import UnpairedFolderDataset
        common = dict(load_size=args.load_size, crop_size=args.crop_size,
                      img_ch=args.img_ch, seed=args.seed, flip=not args.no_flip)
        tr_ds = UnpairedFolderDataset(args.data_root, "train", train=True, **common)
        te_ds = UnpairedFolderDataset(args.data_root, "test", train=False, **common)
    if args.swap_domains:
        # Swap at the source so the losses and the evaluation below are untouched:
        # the "T1"/A slot now holds FA/B and vice versa.
        tr_ds, te_ds = _Swapped(tr_ds), _Swapped(te_ds)
    # Two independent loaders -> unpaired sampling of T1 and FA.
    loader_t1 = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                           num_workers=args.workers, drop_last=True,
                           pin_memory=True)
    loader_fa = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                           num_workers=args.workers, drop_last=True,
                           pin_memory=True)
    print(f"train slices: {len(tr_ds)}  batches/epoch: {len(loader_t1)}", flush=True)
    val_loader = DataLoader(te_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # --- models ---
    C = args.img_ch
    G_T1toFA = init_weights(ResnetGenerator(C, C, args.ngf, args.n_blocks)).to(device)
    G_FAtoT1 = init_weights(ResnetGenerator(C, C, args.ngf, args.n_blocks)).to(device)
    D_FA = init_weights(PatchDiscriminator(C, args.ndf)).to(device)  # judges FA-domain
    D_T1 = init_weights(PatchDiscriminator(C, args.ndf)).to(device)  # judges T1-domain

    # Keep handles on the RAW modules before any wrapping.  Checkpoints must
    # store unwrapped state dicts: every consumer downstream (FlowCycle.
    # load_cyclegan, utils.eval_mnist, utils.make_figures_folder) does a
    # strict=True load into a plain ResnetGenerator, and a "module." prefix
    # from DataParallel would break all of them.
    raw = {"G_T1toFA": G_T1toFA, "G_FAtoT1": G_FAtoT1,
           "D_FA": D_FA, "D_T1": D_T1}
    if args.dp:
        n_gpu = torch.cuda.device_count()
        if n_gpu < 2:
            print(f"--dp requested but only {n_gpu} GPU visible; running single-GPU",
                  flush=True)
        else:
            if args.batch % n_gpu:
                raise SystemExit(f"--batch {args.batch} is not divisible by "
                                 f"{n_gpu} GPUs; DataParallel would give the "
                                 f"cards uneven work")
            print(f"DataParallel over {n_gpu} GPUs "
                  f"({args.batch // n_gpu} images per card)", flush=True)
            G_T1toFA = nn.DataParallel(G_T1toFA)
            G_FAtoT1 = nn.DataParallel(G_FAtoT1)
            D_FA = nn.DataParallel(D_FA)
            D_T1 = nn.DataParallel(D_T1)

    # --- losses ---
    crit_gan = nn.MSELoss()    # LSGAN
    crit_cyc = nn.L1Loss()
    crit_id = nn.L1Loss()

    # --- optimizers ---
    opt_G = torch.optim.Adam(
        itertools.chain(raw["G_T1toFA"].parameters(), raw["G_FAtoT1"].parameters()),
        lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(
        itertools.chain(raw["D_FA"].parameters(), raw["D_T1"].parameters()),
        lr=args.lr, betas=(0.5, 0.999))

    def lr_lambda(epoch):
        if epoch < args.decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - args.decay_start) /
                   float(max(1, args.epochs - args.decay_start)))

    sch_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sch_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    pool_FA = ImagePool(50)
    pool_T1 = ImagePool(50)

    log_path = os.path.join(LOGDIR, "train_log.csv")
    log_f = open(log_path, "w", newline="")
    log_w = csv.writer(log_f)
    log_w.writerow(["epoch", "loss_G", "loss_D", "loss_cyc", "loss_id",
                    "loss_gan", "lr", "sec"])

    def set_requires_grad(nets, flag):
        for net in nets:
            for p in net.parameters():
                p.requires_grad = flag

    n_batches = len(loader_t1)

    for epoch in range(args.epochs):
        t0 = time.time()
        G_T1toFA.train(); G_FAtoT1.train(); D_FA.train(); D_T1.train()
        accum = {k: 0.0 for k in ["G", "D", "cyc", "id", "gan"]}

        for (t1, _fa_unused, _y1), (_t1_unused, fa, _y2) in zip(loader_t1, loader_fa):
            real_T1 = to_minus1_1(t1.to(device, non_blocking=True))
            real_FA = to_minus1_1(fa.to(device, non_blocking=True))

            # ============ Generators ============
            set_requires_grad([D_FA, D_T1], False)
            opt_G.zero_grad()

            # identity
            id_FA = G_T1toFA(real_FA)   # feeding FA into T1->FA should be identity
            id_T1 = G_FAtoT1(real_T1)
            loss_id = (crit_id(id_FA, real_FA) + crit_id(id_T1, real_T1)) * args.lambda_id

            # GAN
            fake_FA = G_T1toFA(real_T1)
            fake_T1 = G_FAtoT1(real_FA)
            pred_fake_FA = D_FA(fake_FA)
            pred_fake_T1 = D_T1(fake_T1)
            valid_FA = torch.ones_like(pred_fake_FA)
            valid_T1 = torch.ones_like(pred_fake_T1)
            loss_gan = crit_gan(pred_fake_FA, valid_FA) + crit_gan(pred_fake_T1, valid_T1)

            # cycle
            rec_T1 = G_FAtoT1(fake_FA)
            rec_FA = G_T1toFA(fake_T1)
            loss_cyc = (crit_cyc(rec_T1, real_T1) + crit_cyc(rec_FA, real_FA)) * args.lambda_cycle

            loss_G = loss_gan + loss_cyc + loss_id
            loss_G.backward()
            opt_G.step()

            # ============ Discriminators ============
            set_requires_grad([D_FA, D_T1], True)
            opt_D.zero_grad()

            # D_FA
            pred_real = D_FA(real_FA)
            loss_D_real = crit_gan(pred_real, torch.ones_like(pred_real))
            fake_FA_ = pool_FA.query(fake_FA.detach())
            pred_fake = D_FA(fake_FA_)
            loss_D_fake = crit_gan(pred_fake, torch.zeros_like(pred_fake))
            loss_D_FA = 0.5 * (loss_D_real + loss_D_fake)
            loss_D_FA.backward()

            # D_T1
            pred_real = D_T1(real_T1)
            loss_D_real = crit_gan(pred_real, torch.ones_like(pred_real))
            fake_T1_ = pool_T1.query(fake_T1.detach())
            pred_fake = D_T1(fake_T1_)
            loss_D_fake = crit_gan(pred_fake, torch.zeros_like(pred_fake))
            loss_D_T1 = 0.5 * (loss_D_real + loss_D_fake)
            loss_D_T1.backward()
            opt_D.step()

            accum["G"] += loss_G.item()
            accum["D"] += (loss_D_FA + loss_D_T1).item()
            accum["cyc"] += loss_cyc.item()
            accum["id"] += loss_id.item()
            accum["gan"] += loss_gan.item()

        sch_G.step(); sch_D.step()
        for k in accum:
            accum[k] /= n_batches
        cur_lr = opt_G.param_groups[0]["lr"]
        dt = time.time() - t0
        print(f"[ep {epoch+1}/{args.epochs}] G={accum['G']:.3f} D={accum['D']:.3f} "
              f"cyc={accum['cyc']:.3f} id={accum['id']:.3f} gan={accum['gan']:.3f} "
              f"lr={cur_lr:.2e} ({dt:.0f}s)", flush=True)
        log_w.writerow([epoch + 1, accum["G"], accum["D"], accum["cyc"],
                        accum["id"], accum["gan"], cur_lr, round(dt, 1)])
        log_f.flush()

        # checkpointing
        def save_ckpt(name):
            torch.save({
                "G_T1toFA": raw["G_T1toFA"].state_dict(),
                "G_FAtoT1": raw["G_FAtoT1"].state_dict(),
                "D_FA": raw["D_FA"].state_dict(),
                "D_T1": raw["D_T1"].state_dict(),
                "args": vars(args),
                "epoch": epoch + 1,
            }, os.path.join(RESULTS, name))

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            save_ckpt(f"ckpt_ep{epoch+1}.pth")
            save_ckpt("last.pth")

        # FAITHFUL (CycleGAN): fixed schedule (const + linear decay), NO early stop,
        # NO checkpoint selection — the reported model is the FINAL one after the full
        # LR-decay schedule (the GAN's generator loss is not a quality signal).

    log_f.close()
    G_T1toFA, G_FAtoT1 = raw["G_T1toFA"], raw["G_FAtoT1"]
    G_T1toFA.eval(); G_FAtoT1.eval()

    if args.data == "folder":
        # No ground-truth pairing: score the two directions distributionally.
        # The reference side is the TRAIN split of the target domain, which is
        # the usual horse2zebra protocol and gives the reference a sample size
        # the 120-image test split cannot.
        import numpy as _np
        from utils import fid as _fid
        from data.unpaired_dataset import list_images as _ls
        R = args.data_root
        cdir = os.path.join(RESULTS, "fid_cache")
        kw = dict(crop=args.crop_size, img_ch=args.img_ch, batch=16)
        A, B = ("B", "A") if args.swap_domains else ("A", "B")   # source, target of G_T1toFA
        f_ab = _fid.fid_against(_ls(f"{R}/test{A}"), G_T1toFA, _ls(f"{R}/train{B}"),
                                device, os.path.join(cdir, f"train{B}.npz"), **kw)
        f_ba = _fid.fid_against(_ls(f"{R}/test{B}"), G_FAtoT1, _ls(f"{R}/train{A}"),
                                device, os.path.join(cdir, f"train{A}.npz"), **kw)
        with open(os.path.join(RESULTS, "final_eval.txt"), "w") as f:
            f.write(f"FINAL (ep{args.epochs})  data={R}  swap_domains={args.swap_domains}\n"
                    f"FID {A}->{B} {f_ab:.2f}\nFID {B}->{A} {f_ba:.2f}\n")
        print(f"training done (full schedule). FID A→B {f_ab:.2f}  B→A {f_ba:.2f} "
              f"-> {RESULTS}/final_eval.txt", flush=True)
        return

    # faithful final-model eval on the FULL test set (one-shot translation)
    import numpy as _np
    from skimage.metrics import structural_similarity as _ssim
    from skimage.metrics import peak_signal_noise_ratio as _psnr
    sAB, sBA, pAB, pBA = [], [], [], []
    with torch.no_grad():
        for t1, fa, _ in val_loader:
            t1 = t1.to(device); fa = fa.to(device)
            fk = to_01(G_T1toFA(to_minus1_1(t1))).cpu().numpy()
            kt = to_01(G_FAtoT1(to_minus1_1(fa))).cpu().numpy()
            gf = fa.cpu().numpy(); gt = t1.cpu().numpy()
            for i in range(fk.shape[0]):
                sAB.append(_ssim(gf[i, 0], fk[i, 0], data_range=1.0))
                sBA.append(_ssim(gt[i, 0], kt[i, 0], data_range=1.0))
                pAB.append(_psnr(gf[i, 0], fk[i, 0], data_range=1.0))
                pBA.append(_psnr(gt[i, 0], kt[i, 0], data_range=1.0))
    with open(os.path.join(RESULTS, "final_eval.txt"), "w") as f:
        if args.swap_domains:
            f.write("swap_domains=True: 'T1toFA' below is the native FA->T1 generator\n")
        f.write(f"FINAL (ep{args.epochs}) T1toFA SSIM {_np.mean(sAB):.4f}±{_np.std(sAB):.4f} "
                f"PSNR {_np.mean(pAB):.2f}\nFA toT1 SSIM {_np.mean(sBA):.4f}±{_np.std(sBA):.4f} "
                f"PSNR {_np.mean(pBA):.2f}\n")
    print(f"training done (full schedule). FINAL-model T1→FA {_np.mean(sAB):.4f} "
          f"FA→T1 {_np.mean(sBA):.4f} (faithful, leakage-free) -> {RESULTS}/final_eval.txt", flush=True)


if __name__ == "__main__":
    main()
