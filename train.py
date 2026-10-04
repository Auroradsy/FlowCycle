#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FlowCycle — three-stage training of the rewired CycleGAN (no GMM).

    S1  two INDEPENDENT autoencoders (+ self-GAN).   train E,D    f frozen = id
    S2  the bijection only.                          train f      E,D frozen
    S3  joint: + cycle, + latent-cycle, + morph.     train f,D    E   frozen

Why S1 does not train the cross paths: if it did, the two feature spaces would
align on their own and the flow would have nothing left to do (flow_work -> 0,
the identity-collapse we measured at 0.34 in the -bNLL run).  Keeping the two
autoencoders independent guarantees f has real work.

Why E stays frozen in S3: the representation is the object of study; letting the
encoders drift while the decoders adapt turns any morph result into a statement
about the decoders instead.

Losses (all GMM / flow-NLL / KL / pair terms are gone):
    L_self    L1(D_A(z), T1) + L1(D_B(u_B), FA)                        x10
    L_cross   L1(D_B(f z), FA) + L1(D_A(f^-1 u_B), T1)                 x10   <- paired data
    L_gan     LSGAN, CycleGAN discriminators                               x1
    L_cyc     pixel cycle T1->FA->T1                                   x10
    L_latcyc  || E_B(D_B(f z)) - f z ||_1 / || f z ||_1                x1..5
    L_path    LSGAN(D_mix(D_B(s_t))) + || D_B(s_t) - D_B(s_{t-1}) ||_1

`pair` is deliberately absent: across the flow3 runs latent alignment and
cross-modal SSIM were anti-correlated (gap 9.07 -> 0.77 -> 0.091 while cross
went 0.749 -> 0.706 -> 0.738).

L_latcyc is the term that was missing from all eight earlier morph attempts: it
constrains the flow's output to be a FIXED POINT of the B autoencoder, i.e. to
be decodable, without constraining its distribution (which is what -bNLL did,
at the cost of collapsing the flow's work to 0.34).

  python train.py --config configs/morph.yaml
  bash configs/run_morph.sh
"""
import os
import sys
import csv
import time
import random
import argparse
import itertools

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data.paired_dataset import (                                        # noqa: E402
    build_cache, PairedADNISliceDataset, subject_level_split, CACHE)
from model import FlowCycle, make_discriminators                         # noqa: E402
from utils.image import to_pm1, to_01                                    # noqa: E402
from utils.pool import ImagePool                                         # noqa: E402
from utils.config import parse_with_config                               # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Everything a run produces lives under exps/ (gitignored).  Override the whole
# tree with FLOWCYCLE_EXPS to keep several experiment sets side by side.
from server_paths import experiment_root, checkpoint_root
EXPS = experiment_root()
CKPT = os.path.join(EXPS, "checkpoints")
LOGS = os.path.join(EXPS, "logs")
CFM_T1FA, CFM_FAT1 = 0.729, 0.790          # best baseline, RECON_RESULTS.md §1


def val_split_from_train(train_idx, seed=42, frac=0.15):
    """Carve a validation set out of the TRAIN subjects (leakage-free)."""
    d = torch.load(CACHE, map_location="cpu")
    subj = d["subj_idx"].numpy()[train_idx.numpy()]
    uniq = np.unique(subj)
    perm = np.random.default_rng(seed).permutation(len(uniq))
    n = max(1, int(round(frac * len(uniq))))
    va_s = set(uniq[perm[:n]].tolist())
    mask = np.array([s in va_s for s in subj])
    return train_idx[~mask], train_idx[mask]


# ---------------------------------------------------------------------------
def _ssim(g, p):
    """SSIM between two (C,H,W) arrays in [0,1]; colour images score jointly."""
    from skimage.metrics import structural_similarity as ssim
    if g.shape[0] == 1:
        return ssim(g[0], p[0], data_range=1.0)
    return ssim(g.transpose(1, 2, 0), p.transpose(1, 2, 0),
                data_range=1.0, channel_axis=2)


@torch.no_grad()
def evaluate(m, loader, shuffle_probe=False, paired=True):
    """SSIM for the four paths + latent diagnostics (+ optional u-shuffle).

    With `paired=False` the loader's two images are unrelated, so every metric
    that reads one as ground truth for the other is dropped rather than reported
    as a meaningless number: only the two self-reconstructions, flow_work and
    latent_gap survive.  The u-shuffle probe is replaced by `shuf_delta`, the
    mean absolute change in the OUTPUT when the code is shuffled — same question
    ("is the latent load-bearing?") asked without a ground-truth target.
    """
    m.eval()
    acc = {k: [] for k in ["self_T1", "self_FA", "T1toFA", "FAtoT1",
                           "shuf_FA", "shuf_T1", "ref_FA", "ref_T1"]}
    fw, gap, sd = [], [], []
    for t1, fa, _ in loader:
        T = to_pm1(t1.to(DEV)); F = to_pm1(fa.to(DEV))
        zA = m.enc_A(T); uB = m.enc_B(F)
        u = m.a_to_b(zA); z = m.b_to_a(uB)
        im = {"self_T1": (to_01(m.dec_A(zA)), t1), "self_FA": (to_01(m.dec_B(uB)), fa)}
        if paired:
            im["T1toFA"] = (to_01(m.dec_B(u)), fa)
            im["FAtoT1"] = (to_01(m.dec_A(z)), t1)
        if shuffle_probe and u.shape[0] > 1:
            r = torch.roll(torch.arange(u.shape[0]), 1)
            if paired:
                im["shuf_FA"] = (to_01(m.dec_B(u[r])), fa)
                im["shuf_T1"] = (to_01(m.dec_A(z[r])), t1)
                # Calibration: every slice here is MNI-registered, so two DIFFERENT
                # real brains already score high against each other.  ref_* is that
                # floor — the score a perfect model gets when handed the wrong code.
                # A load-bearing latent lands ON this floor; a decorative one (the
                # FiLM head of §3, Δ = 0.000) stays at its unshuffled score.
                im["ref_FA"] = (fa[r], fa); im["ref_T1"] = (t1[r], t1)
            else:
                sd.append(float((m.dec_B(u[r]) - m.dec_B(u)).abs().mean()))
        for k, (pred, gt) in im.items():
            p = pred.cpu().numpy() if pred.is_cuda else pred.numpy()
            g = gt.numpy()
            for i in range(p.shape[0]):
                acc[k].append(_ssim(g[i], p[i]))
        fw.append(m.flow_work(zA)); gap.append(m.latent_gap(zA, uB))
    out = {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}
    out["flow_work"] = float(np.mean(fw)); out["latent_gap"] = float(np.mean(gap))
    out["shuf_delta"] = float(np.mean(sd)) if sd else float("nan")
    m.train()
    return out


@torch.no_grad()
def val_monitor(m, loader, stage, w_latcyc, paired=True):
    """Non-adversarial validation loss — the early-stopping criterion.

    loss_gan oscillates by construction, so it is excluded (same rule as the
    faithful CycleGAN runs).

    Paired data lets S2/S3 be scored by L_cross, the term they are actually
    minimising.  Unpaired, that term does not exist, so the criterion becomes
    the non-adversarial part of the unpaired objective instead: self
    reconstruction, the pixel cycle, and the latent cycle.  Both versions are
    "the trainable objective minus the GAN", which is what makes early stopping
    mean the same thing in either mode -- with one deliberate exception in the
    unpaired branch, documented at the `sel` line below.
    """
    m.eval(); tot, n = 0.0, 0
    l1 = nn.L1Loss()
    for t1, fa, _ in loader:
        T = to_pm1(t1.to(DEV)); F = to_pm1(fa.to(DEV))
        zA = m.enc_A(T); uB = m.enc_B(F)
        v = 0.0
        if stage in (1, 3):
            v = v + l1(m.dec_A(zA), T).item() + l1(m.dec_B(uB), F).item()
        if stage in (2, 3):
            u = m.a_to_b(zA); z = m.b_to_a(uB)
            fake_FA = m.dec_B(u); fake_T1 = m.dec_A(z)
            if paired:
                v = v + l1(fake_FA, F).item() + l1(fake_T1, T).item()
                if stage == 3:
                    back = m.enc_B(fake_FA)
                    v = v + w_latcyc * float((back - u).abs().mean() /
                                             (u.abs().mean() + 1e-8))
            else:
                back_u = m.enc_B(fake_FA); back_z = m.enc_A(fake_T1)
                v = v + l1(m.dec_A(m.b_to_a(back_u)), T).item() \
                      + l1(m.dec_B(m.a_to_b(back_z)), F).item()
                # The pixel cycle above is very nearly BLIND to f, and that
                # blindness cost a whole horse2zebra arm.  At S2 the encoders
                # and decoders are frozen, so enc_B.dec_B ~ I, and f is an
                # exact bijection, so b_to_a(a_to_b(.)) cancels analytically:
                # the term collapses to dec_A(enc_A(T)) plus second-order
                # residue no matter what f does.  With w_latcyc = 0 nothing
                # else in the criterion touched f either, so `h2z_base`
                # "converged" at ep5 and patience killed S2 at ep20 -- FID
                # 226.2, i.e. sitting on the do-nothing ceiling.  Re-running
                # that arm with the stop removed gave 109.6, the best of any
                # arm: the objective was fine, the criterion was not.
                #
                # So the latent-cycle residual is always included here, even
                # when it carries no weight in the objective.  It is the only
                # quantity available at S2 that depends on f to FIRST order,
                # and it measures the right thing: back_u - u is the
                # autoencoder error evaluated at f(z_A), which is small exactly
                # when f lands inside the range D_B was trained to decode.
                # This is a deliberate break from "criterion = objective minus
                # GAN" -- a criterion that cannot see the parameters being
                # trained cannot select among them.
                # The fallback below applies to S2 ONLY.  At S2 the encoders
                # and decoders are frozen, so the self term is gated out of the
                # criterion entirely and the latent-cycle residual really is
                # the only first-order-sensitive quantity left.  At S3 the
                # decoders train and `self` tracks them directly, so the
                # criterion is not blind -- and forcing the residual in there
                # at unit weight actively hurts: for a w_latcyc = 0 arm it is
                # ~96% of the value (base's S3 val 2.073 = 1.983 latcyc + 0.011
                # self x2 + cycle), so it drowns out the one term that was
                # improving.  That is what cut mnist_base's S3 at ep16 while
                # its self loss was still falling, 0.011 -> 0.006.
                sel = w_latcyc if (w_latcyc > 0 or stage != 2) else 1.0
                v = v + sel * float(
                    (back_u - u).abs().mean() / (u.abs().mean() + 1e-8)
                    + (back_z - z).abs().mean() / (z.abs().mean() + 1e-8))
        tot += v; n += 1
    m.train()
    return tot / max(n, 1)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="FlowCycle three-stage trainer.  See configs/ for ready-made runs.")
    ap.add_argument("--config", default=None,
                    help="YAML of hyperparameters; command-line flags still win")
    ap.add_argument("--variant", choices=["base", "latcyc", "morph", "morph_bi"],
                    default="morph")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--warm", default=str(checkpoint_root() / "cyclegan" / "last.pth"),
                    help="plain-CycleGAN checkpoint to split into E/D ('' = from scratch)")
    # data
    ap.add_argument("--z_lo", type=int, default=40, help="axial band low, inclusive")
    ap.add_argument("--z_hi", type=int, default=49, help="axial band high, inclusive")
    ap.add_argument("--label_scheme", default="label_4")
    ap.add_argument("--resume_stage", type=int, default=0,
                    help="skip stages <= this and load weights from --resume_from. "
                         "Use 2 to share one S1+S2 across variants so that S3 is "
                         "the only thing that differs.")
    ap.add_argument("--resume_from", default="",
                    help="stage checkpoint to resume from (default: this tag's own)")
    ap.add_argument("--check_init", action="store_true",
                    help="verify f=id + the E/D split reproduces the CycleGAN, then exit")
    # schedule
    ap.add_argument("--e1", type=int, default=120); ap.add_argument("--p1", type=int, default=20)
    ap.add_argument("--e2", type=int, default=80);  ap.add_argument("--p2", type=int, default=15)
    ap.add_argument("--e3", type=int, default=120); ap.add_argument("--p3", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_flow", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=5.0)
    # architecture
    ap.add_argument("--ngf", type=int, default=64); ap.add_argument("--ndf", type=int, default=64)
    ap.add_argument("--n_blocks", type=int, default=6)
    ap.add_argument("--n_flow", type=int, default=4)
    ap.add_argument("--flow_hidden", type=int, default=128)
    ap.add_argument("--pre_relu", type=int, default=1)
    ap.add_argument("--shared_dec", type=int, default=0,
                    help="1 = ONE decoder for both modalities (dec_B is dec_A); "
                         "incompatible with --warm, which carries a decoder per direction")
    # loss weights
    ap.add_argument("--w_self", type=float, default=10.0)
    ap.add_argument("--w_cross", type=float, default=10.0)
    ap.add_argument("--w_gan", type=float, default=1.0)
    ap.add_argument("--w_cyc", type=float, default=10.0)
    ap.add_argument("--w_latcyc", type=float, default=2.0)
    ap.add_argument("--w_path_gan", type=float, default=0.5)
    ap.add_argument("--w_path_smooth", type=float, default=1.0)
    ap.add_argument("--path_gan_mode", default="abs", choices=["abs", "ra"],
                    help="abs = LSGAN, intermediates targeted at 'real' (unreachable "
                         "by construction: an in-between frame is in neither real set). "
                         "ra = relativistic LSGAN, they only need to rank near the reals.")
    ap.add_argument("--path_critics", default="shared", choices=["shared", "separate"],
                    help="with path_bidir, give the B->A leg its own critic")
    ap.add_argument("--path_bidir", type=int, default=0,
                    help="also supervise the B->A path (walk f backwards, read with D_A)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)

    # --- unpaired image folders (CycleGAN layout) -------------------------
    # `adni` is the original path and behaves exactly as before.  `folder`
    # swaps in <root>/{trainA,trainB,testA,testB}, where the two images in a
    # batch are UNRELATED.  That makes L_cross meaningless, so it must be
    # switched off (--w_cross 0) — see the guard below.
    ap.add_argument("--data", default="adni", choices=["adni", "folder"])
    ap.add_argument("--data_root", default="")
    ap.add_argument("--img_ch", type=int, default=1)
    ap.add_argument("--load_size", type=int, default=286)
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--pair", action="store_true",
                    help="the image folders are PAIRED: <split>A/x.png and "
                         "<split>B/x.png are the same sample.  Enables L_cross "
                         "and SSIM-against-truth, i.e. the ADNI protocol on "
                         "folder data.  Alignment is asserted at load time.")
    ap.add_argument("--no_flip", action="store_true",
                    help="disable h-flip augmentation; required for chiral "
                         "content such as digits, where a mirrored sample is "
                         "not a valid member of the domain")
    ap.add_argument("--eval_batch", type=int, default=8)
    ap.add_argument("--s2_cyc", type=int, default=0,
                    help="also apply L_cyc/L_latcyc in stage 2.  Required when "
                         "w_cross=0: S2 otherwise trains the flow on the "
                         "adversarial term alone, with nothing tying f's output "
                         "to the range of E_B.")
    ap.add_argument("--fid_every", type=int, default=0,
                    help="epochs between FID evaluations (0 = only at the end). "
                         "Unpaired runs have no SSIM-vs-truth, so this is the "
                         "only quality number during training.")
    a = parse_with_config(ap)

    # A config may give `warm` as a repo-relative path; resolve it here so the
    # script works from any working directory, not just the repo root.
    if a.warm and not os.path.isabs(a.warm):
        from server_paths import resolve_checkpoint
        a.warm = resolve_checkpoint(a.warm)
        if not os.path.isabs(a.warm):
            a.warm = os.path.join(_HERE, a.warm)

    # --warm has a default path, so a shared-decoder run that forgot to clear it would
    # warm-start into the overwrite load_cyclegan refuses. Say so before anything loads.
    if a.shared_dec and a.warm:
        raise SystemExit("--shared_dec needs --warm '' : one decoder cannot be initialised "
                         "from a CycleGAN, which has one per direction")

    if a.variant == "base":
        a.w_latcyc = 0.0; a.w_path_gan = 0.0; a.w_path_smooth = 0.0; a.path_bidir = 0
    elif a.variant == "latcyc":
        a.w_path_gan = 0.0; a.w_path_smooth = 0.0; a.path_bidir = 0
    elif a.variant == "morph":
        a.path_bidir = 0
    elif a.variant == "morph_bi":
        a.path_bidir = 1
    use_path = a.w_path_gan > 0 or a.w_path_smooth > 0
    # `paired` selects the PROTOCOL (L_cross, SSIM-vs-truth, the val criterion);
    # `a.data` selects the LOADER.  They used to be the same switch, which made
    # a paired image-folder dataset unexpressible.
    paired = a.data == "adni" or a.pair
    if a.data != "adni":
        if not a.data_root:
            raise SystemExit("--data folder requires --data_root")
        if a.w_cross > 0 and not a.pair:
            raise SystemExit(
                "--data folder is UNPAIRED: L_cross would L1-match two unrelated "
                "images and collapse both decoders to the domain mean.  Pass "
                "--w_cross 0 (and --s2_cyc 1, or stage 2 has only the GAN term).")
        if not a.s2_cyc and not a.pair:
            print("WARNING: w_cross=0 and --s2_cyc 0 — stage 2 trains the flow on "
                  "the adversarial term alone.  Expect flow_work to stay near 0.",
                  flush=True)
    tag = a.tag or a.variant
    RESULTS = os.path.join(CKPT, tag)                 # weights + final_eval.txt
    LOGDIR = os.path.join(LOGS, tag)                  # per-epoch csv
    os.makedirs(RESULTS, exist_ok=True); os.makedirs(LOGDIR, exist_ok=True)

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    print(f"device={DEV}  variant={a.variant}\n  weights -> {RESULTS}\n  log     -> {LOGDIR}\nargs={vars(a)}", flush=True)

    # --- data -------------------------------------------------------------
    if a.data == "adni":
        build_cache()
        tr_idx, te_idx = subject_level_split(a.seed, 0.20, a.label_scheme, a.z_lo, a.z_hi)
        tr_idx, va_idx = val_split_from_train(tr_idx, a.seed, 0.15)
        tl = DataLoader(PairedADNISliceDataset(tr_idx, a.label_scheme), batch_size=a.batch,
                        shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True)
        vl = DataLoader(PairedADNISliceDataset(va_idx, a.label_scheme), batch_size=32,
                        shuffle=False, num_workers=2)
        el = DataLoader(PairedADNISliceDataset(te_idx, a.label_scheme), batch_size=32,
                        shuffle=False, num_workers=2)
        print(f"train {len(tr_idx)} / val {len(va_idx)} / test {len(te_idx)} slices  "
              f"({len(tl)} batches/epoch)", flush=True)
    else:
        from data.unpaired_dataset import folder_loaders
        tl, vl, el, n_tr, n_va, n_te = folder_loaders(
            a.data_root, a.batch, a.workers, a.load_size, a.crop_size,
            a.img_ch, a.seed, eval_batch=a.eval_batch, flip=not a.no_flip,
            pair=a.pair)
        print(f"train {n_tr} / val {n_va} / test {n_te} images from {a.data_root}  "
              f"({len(tl)} batches/epoch)", flush=True)

    # --- init check: does the split + f=id reproduce plain CycleGAN? ------
    if a.check_init:
        mm = FlowCycle(a.ngf, a.n_blocks, a.n_flow, a.flow_hidden, pre_relu=False,
                       img_ch=a.img_ch).to(DEV)
        mm.load_cyclegan(a.warm, map_location=DEV)
        r = evaluate(mm, el, paired=paired)
        print(f"\n[check_init] pre_relu=0, f=identity")
        if paired:
            # derive from --warm rather than hard-coding the ADNI CycleGAN, so a
            # paired folder run cites its own CycleGAN (for ADNI this is unchanged)
            ref = os.path.join(os.path.dirname(a.warm), "final_eval.txt")
            print(f"  T1->FA {r['T1toFA']:.4f}   FA->T1 {r['FAtoT1']:.4f}")
            print(f"  flow_work {r['flow_work']:.2e} (must be ~0)")
            if os.path.exists(ref):
                print("  CycleGAN reference:\n   ",
                      open(ref).read().replace("\n", "\n    ").strip())
            return
        # Unpaired: there is no SSIM-vs-truth to compare against the CycleGAN, so the
        # equivalence is checked directly — the split model's cross paths must be
        # the CycleGAN generators to floating-point noise.
        print(f"  flow_work {r['flow_work']:.2e} (must be ~0)")
        from model.backbone import ResnetGenerator
        ck = torch.load(a.warm, map_location=DEV)
        worst = 0.0
        mm.eval()
        for key, fn, side in [("G_T1toFA", mm.cross_A2B, 0),
                              ("G_FAtoT1", mm.cross_B2A, 1)]:
            g = ResnetGenerator(a.img_ch, a.img_ch, a.ngf, a.n_blocks).to(DEV).eval()
            g.load_state_dict(ck[key], strict=True)
            worst_k = 0.0
            with torch.no_grad():
                for batch in el:
                    inp = to_pm1(batch[side].to(DEV))
                    worst_k = max(worst_k, float((fn(inp) - g(inp)).abs().max()))
            print(f"  max |split({key}) - cyclegan({key})| = {worst_k:.3e}")
            worst = max(worst, worst_k)
        print(f"  -> {'PASS' if worst < 1e-4 else 'FAIL'} (threshold 1e-4)")
        return

    # --- model ------------------------------------------------------------
    m = FlowCycle(a.ngf, a.n_blocks, a.n_flow, a.flow_hidden, bool(a.pre_relu),
                  img_ch=a.img_ch, shared_dec=bool(a.shared_dec)).to(DEV)
    if a.shared_dec:
        print("shared decoder: dec_B IS dec_A, one decoder for both modalities", flush=True)
    if a.warm and os.path.exists(a.warm):
        ep = m.load_cyclegan(a.warm, map_location=DEV)
        print(f"warm start from {a.warm} (CycleGAN epoch {ep})", flush=True)
    else:
        print("no warm start — training the CycleGAN from scratch", flush=True)
    D = make_discriminators(a.ndf, mix=use_path,
                            mix_b=bool(use_path and a.path_bidir
                                       and a.path_critics == "separate"),
                            img_ch=a.img_ch)
    for k in D:
        D[k] = D[k].to(DEV)

    if a.resume_stage:
        from server_paths import resolve_checkpoint, previous_checkpoint
        src = resolve_checkpoint(a.resume_from) if a.resume_from else previous_checkpoint(a.resume_stage)
        if not os.path.isabs(src):
            src = os.path.join(_HERE, src)
        ck = torch.load(src, map_location=DEV)
        m.load_state_dict(ck["model"])
        # Stage checkpoints written before this flag existed carry no
        # discriminators; those runs restart S3 with fresh critics.  Say so out
        # loud — it is a real difference between two otherwise-identical runs,
        # so every arm of a comparison must resume the same way.
        have_d = [k for k in D if f"d_{k}" in ck]
        for k in have_d:
            D[k].load_state_dict(ck[f"d_{k}"])
        print(f"resumed from {src} (through stage {ck.get('stage')}); "
              f"discriminators: {'restored ' + ','.join(have_d) if have_d else 'FRESH (not in checkpoint)'}",
              flush=True)

    # with a shared decoder dec_A IS dec_B, so both the count and the optimizer group
    # must see each tensor once: Adam would otherwise take two steps per update on the
    # decoder, and the printed size would be a decoder too large.
    def uniq(mods):
        seen, out = set(), []
        for mod in mods:
            for p in mod.parameters():
                if id(p) not in seen:
                    seen.add(id(p)); out.append(p)
        return out

    cyclegan_params = uniq((m.enc_A, m.enc_B, m.dec_A, m.dec_B))
    n_cyclegan = sum(p.numel() for p in cyclegan_params)
    print(f"params: CycleGAN {n_cyclegan/1e6:.2f} M + flow {sum(p.numel() for p in m.flow.parameters())/1e6:.2f} M",
          flush=True)

    opt_G = torch.optim.Adam([
        {"params": cyclegan_params, "lr": a.lr},
        {"params": m.flow.parameters(), "lr": a.lr_flow}], betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(itertools.chain(*[D[k].parameters() for k in D]),
                             lr=a.lr, betas=(0.5, 0.999))
    crit_gan, l1 = nn.MSELoss(), nn.L1Loss()
    pools = {k: ImagePool(50) for k in D}

    def req(mods, flag):
        for mod in mods:
            for p in mod.parameters():
                p.requires_grad = flag

    def set_stage(s):
        enc, dec, flow = [m.enc_A, m.enc_B], [m.dec_A, m.dec_B], [m.flow]
        if s == 1:   req(enc, True);  req(dec, True);  req(flow, False)
        elif s == 2: req(enc, False); req(dec, False); req(flow, True)
        else:        req(enc, False); req(dec, True);  req(flow, True)
        print(f"  [stage{s}] trainable: enc={s==1} dec={s!=2} flow={s!=1}", flush=True)

    def snapshot():
        return {k: {kk: vv.detach().cpu().clone() for kk, vv in mod.state_dict().items()}
                for k, mod in [("m", m)] + [(f"d_{k}", D[k]) for k in D]}

    def restore(sn):
        m.load_state_dict(sn["m"])
        for k in D:
            D[k].load_state_dict(sn[f"d_{k}"])

    log = open(os.path.join(LOGDIR, "train_log.csv"), "w", newline="")
    W = csv.writer(log)
    W.writerow(["epoch", "stage", "G", "D", "self", "cross", "gan", "cyc", "latcyc",
                "path_gan", "path_sm", "val", "self_T1", "self_FA", "T1toFA", "FAtoT1",
                "flow_work", "latent_gap", "sec"])

    ep_global = 0
    stopped = False
    stopfile = os.path.join(RESULTS, "STOP")     # `touch` it to end the run cleanly
    if os.path.exists(stopfile):
        os.remove(stopfile)                      # a leftover must not kill a new run
    for stage, n_ep, patience in [(1, a.e1, a.p1), (2, a.e2, a.p2), (3, a.e3, a.p3)]:
        if stage <= a.resume_stage:
            print(f"  [stage{stage}] skipped (resumed)", flush=True)
            continue
        set_stage(stage)
        if stage == 2:                       # the flow alone gets the full LR
            opt_G.param_groups[1]["lr"] = a.lr_flow * 2
        if stage == 3:
            opt_G.param_groups[0]["lr"] = a.lr * 0.5
            opt_G.param_groups[1]["lr"] = a.lr_flow
        best, bad, best_sn = float("inf"), 0, None

        for e in range(n_ep):
            t0 = time.time(); m.train()
            acc = dict(G=0., D=0., self=0., cross=0., gan=0., cyc=0.,
                       latcyc=0., path_gan=0., path_sm=0.); nb = 0

            for t1, fa, _ in tl:
                T = to_pm1(t1.to(DEV, non_blocking=True))
                F = to_pm1(fa.to(DEV, non_blocking=True))
                nb += 1
                req(list(D.values()), False); opt_G.zero_grad()

                zA = m.enc_A(T); uB = m.enc_B(F)
                real_mix = torch.cat([T, F], 0)     # D_mix's real set: T1 U FA
                fakes = {}                         # name -> (image, discriminator key)
                L_self = torch.zeros((), device=DEV)
                L_cross = torch.zeros((), device=DEV)
                L_cyc = torch.zeros((), device=DEV)
                L_lat = torch.zeros((), device=DEV)
                L_pg = torch.zeros((), device=DEV)
                L_ps = torch.zeros((), device=DEV)

                if stage in (1, 3):
                    rec_A = m.dec_A(zA); rec_B = m.dec_B(uB)
                    L_self = l1(rec_A, T) + l1(rec_B, F)
                    fakes["self_T1"] = (rec_A, "T1"); fakes["self_FA"] = (rec_B, "FA")

                if stage in (2, 3):
                    u = m.a_to_b(zA); z = m.b_to_a(uB)
                    fake_FA = m.dec_B(u); fake_T1 = m.dec_A(z)
                    if a.w_cross > 0:            # paired supervision only
                        L_cross = l1(fake_FA, F) + l1(fake_T1, T)
                    fakes["cross_FA"] = (fake_FA, "FA"); fakes["cross_T1"] = (fake_T1, "T1")

                    # S2 normally rides on L_cross; with it gone, the cycle and
                    # latent-cycle terms are what give the flow an objective
                    # beyond fooling the critics.  E and D are still frozen, so
                    # the gradient reaches f alone.
                    if stage == 3 or a.s2_cyc:
                        # cycle T1->FA->T1 and FA->T1->FA.  The flow cancels
                        # analytically ONLY if E_B(D_B(.)) = id, so this is a real
                        # constraint on the encoder/decoder pair.
                        back_u = m.enc_B(fake_FA); back_z = m.enc_A(fake_T1)
                        rec_T1 = m.dec_A(m.b_to_a(back_u))
                        rec_FA = m.dec_B(m.a_to_b(back_z))
                        L_cyc = l1(rec_T1, T) + l1(rec_FA, F)
                        # latent cycle: f's output must be a fixed point of the
                        # B autoencoder (scale-free, so it cannot be minimised by
                        # shrinking the code).
                        L_lat = ((back_u - u).abs().mean() / (u.abs().mean() + 1e-8)
                                 + (back_z - z).abs().mean() / (z.abs().mean() + 1e-8))

                    if stage == 3 and use_path:
                        # Trajectory supervision: one random adjacent pair per
                        # step, per supervised direction.
                        #
                        # A->B walks f forward and reads the states with D_B;
                        # B->A walks f backward and reads them with D_A.  With
                        # path_bidir=0 only the first exists, which is why every
                        # FA->T1 intermediate is unsupervised (fig 33).
                        #
                        # Both halves are AVERAGED, not summed, so w_path_* keeps
                        # the same effective scale in either mode and the
                        # ablation is about direction rather than weight.
                        legs = [(m.walk(zA), m.dec_B, "mix")]
                        if a.path_bidir:
                            legs.append((m.walk(uB, inverse=True), m.dec_A,
                                         "mix_b" if "mix_b" in D else "mix"))

                        by_leg = {}
                        for states, dec, dk in legs:
                            t = random.randint(1, len(states) - 1)
                            prev = dec(states[t - 1]); cur = dec(states[t])
                            L_ps = L_ps + (cur - prev).abs().mean()
                            by_leg.setdefault(dk, []).extend([prev, cur])
                        L_ps = L_ps / len(legs)

                        if a.w_path_gan > 0:
                            for dk, imgs in by_leg.items():
                                if a.path_gan_mode == "ra":
                                    pf = D[dk](torch.cat(imgs, 0)); pr = D[dk](real_mix)
                                    # x2 so the nominal scale matches `abs`, which
                                    # sums one term per image (2 per leg).
                                    L_pg = L_pg + 2.0 * 0.5 * (
                                        ((pf - pr.mean() - 1) ** 2).mean()
                                        + ((pr - pf.mean() + 1) ** 2).mean())
                                else:
                                    for img in imgs:
                                        pp = D[dk](img)
                                        L_pg = L_pg + crit_gan(pp, torch.ones_like(pp))
                            L_pg = L_pg / len(legs)
                            for dk, imgs in by_leg.items():
                                fakes[f"path_{dk}"] = (torch.cat(imgs, 0), dk)

                L_gan = torch.zeros((), device=DEV)
                for _k, (img, dk) in fakes.items():
                    if dk == "mix":
                        continue                       # already counted in L_pg
                    p = D[dk](img)
                    L_gan = L_gan + crit_gan(p, torch.ones_like(p))

                loss_G = (a.w_self * L_self + a.w_cross * L_cross + a.w_gan * L_gan
                          + a.w_cyc * L_cyc + a.w_latcyc * L_lat
                          + a.w_path_gan * L_pg + a.w_path_smooth * L_ps)
                loss_G.backward()
                if a.clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in opt_G.param_groups for p in g["params"]
                         if p.requires_grad], a.clip)
                opt_G.step()

                # ---- discriminators ----
                req(list(D.values()), True); opt_D.zero_grad()
                real = {"T1": T, "FA": F, "mix": real_mix, "mix_b": real_mix}
                by_d = {}
                for _k, (img, dk) in fakes.items():
                    by_d.setdefault(dk, []).append(img.detach())
                d_tot = 0.0
                for dk, imgs in by_d.items():
                    pr = D[dk](real[dk])
                    pf = D[dk](pools[dk].query(torch.cat(imgs, 0)))
                    if dk.startswith("mix") and a.path_gan_mode == "ra":
                        ld = 0.5 * (((pr - pf.mean() - 1) ** 2).mean()
                                    + ((pf - pr.mean() + 1) ** 2).mean())
                    else:
                        ld = 0.5 * (crit_gan(pr, torch.ones_like(pr))
                                    + crit_gan(pf, torch.zeros_like(pf)))
                    ld.backward(); d_tot += ld.item()
                opt_D.step()

                for k, v in [("G", loss_G.item()), ("D", d_tot), ("self", float(L_self)),
                             ("cross", float(L_cross)), ("gan", float(L_gan)),
                             ("cyc", float(L_cyc)), ("latcyc", float(L_lat)),
                             ("path_gan", float(L_pg)), ("path_sm", float(L_ps))]:
                    acc[k] += v

            acc = {k: v / max(nb, 1) for k, v in acc.items()}
            vloss = val_monitor(m, vl, stage, a.w_latcyc, paired=paired)
            ep_global += 1
            row = [ep_global, stage] + [round(acc[k], 4) for k in
                                        ["G", "D", "self", "cross", "gan", "cyc",
                                         "latcyc", "path_gan", "path_sm"]] + [round(vloss, 5)]

            if (e + 1) % 10 == 0 or e == n_ep - 1:
                r = evaluate(m, el, paired=paired)
                # Unpaired runs have no cross-SSIM; flow_work carries the load
                # instead, since identity collapse is the failure mode that
                # dropping L_cross exposes.
                cross = (f"T1→FA {r['T1toFA']:.3f} FA→T1 {r['FAtoT1']:.3f}"
                         if paired else f"cyc={acc['cyc']:.3f}")
                print(f"[s{stage} ep{e+1}/{n_ep}] G={acc['G']:.3f} self={acc['self']:.3f} "
                      f"cross={acc['cross']:.3f} lat={acc['latcyc']:.3f} val={vloss:.4f} | "
                      f"selfA {r['self_T1']:.3f} selfB {r['self_FA']:.3f} | {cross} | "
                      f"fw={r['flow_work']:.2f} gap={r['latent_gap']:.2f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                row += [round(r[k], 4) for k in ["self_T1", "self_FA", "T1toFA", "FAtoT1",
                                                 "flow_work", "latent_gap"]]
            else:
                print(f"[s{stage} ep{e+1}/{n_ep}] G={acc['G']:.3f} self={acc['self']:.3f} "
                      f"cross={acc['cross']:.3f} lat={acc['latcyc']:.3f} "
                      f"path={acc['path_gan']:.3f}/{acc['path_sm']:.3f} val={vloss:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                row += ["", "", "", "", "", ""]
            row.append(round(time.time() - t0, 1)); W.writerow(row); log.flush()

            if vloss < best - 1e-5:
                best, bad, best_sn = vloss, 0, snapshot()
            else:
                bad += 1
            if bad >= patience:
                print(f"  [stage{stage}] early stop @ep{e+1} (best val {best:.5f})", flush=True)
                break
            # Graceful stop.  Nothing is written to disk until a stage ends, so
            # SIGKILL on a run that is 80 epochs into S3 throws all of it away —
            # the best-val snapshot lives only in memory.  `touch <run>/STOP`
            # instead: the current stage finishes the way an early stop would
            # (restore best, write stageN.pth), the remaining stages are
            # skipped, and the final model + eval are still produced.
            if os.path.exists(stopfile):
                print(f"  [stage{stage}] STOP requested @ep{e+1} "
                      f"(best val {best:.5f}) — finishing up", flush=True)
                stopped = True
                break

        if best_sn is not None:
            restore(best_sn)
            print(f"  [stage{stage}] restored best (val {best:.5f})", flush=True)
        snap = dict(model=m.state_dict(), args=vars(a), stage=stage)
        snap.update({f"d_{k}": D[k].state_dict() for k in D})   # so a resume is exact
        torch.save(snap, os.path.join(RESULTS, f"stage{stage}.pth"))
        if stopped:
            os.remove(stopfile)
            print(f"  skipping the remaining stages; scoring the stage-{stage} model",
                  flush=True)
            break

    # --- final ------------------------------------------------------------
    torch.save(dict(model=m.state_dict(), args=vars(a)), os.path.join(RESULTS, "model.pth"))
    r = evaluate(m, el, shuffle_probe=True, paired=paired)

    if a.data != "adni" and not paired:
        # Distributional scoring, three numbers:
        #   endpoint FID  — the usual A->B / B->A quality, comparable to the CycleGAN
        #   path FID      — intermediate frames against real A U real B, the
        #                   unpaired stand-in for the hole metric
        # The reference side is the target domain's TRAIN split (the standard
        # horse2zebra protocol; the test split is too small to be a reference).
        from utils import fid as F
        from data.unpaired_dataset import list_images
        m.eval()
        R, cdir = a.data_root, os.path.join(RESULTS, "fid_cache")
        kw = dict(crop=a.crop_size, img_ch=a.img_ch, batch=a.eval_batch)
        f_ab = F.fid_against(list_images(f"{R}/testA"), m.cross_A2B,
                             list_images(f"{R}/trainB"), DEV,
                             os.path.join(cdir, "trainB.npz"), **kw)
        f_ba = F.fid_against(list_images(f"{R}/testB"), m.cross_B2A,
                             list_images(f"{R}/trainA"), DEV,
                             os.path.join(cdir, "trainA.npz"), **kw)
        both = sorted(list_images(f"{R}/trainA") + list_images(f"{R}/trainB"))
        mu_m, s_m = F.folder_stats(both, DEV, os.path.join(cdir, "train_AB.npz"),
                                   crop=a.crop_size)
        n_states = a.n_flow + 1
        path_fid = []
        for t in range(n_states):
            fn = (lambda x, t=t: m.dec_B(m.walk(m.enc_A(x))[t]))
            mu_f, s_f = F.translated_stats(list_images(f"{R}/testA"), fn, DEV, **kw)
            path_fid.append(F.frechet(mu_f, s_f, mu_m, s_m))
        m.train()
        # Sample sizes matter for reading these: FID is biased upward at small
        # n, so the frame-to-frame comparison (same n throughout) is the signal,
        # and the absolute value is only loosely comparable to the literature.
        txt = (f"variant={a.variant}\ndata={R}\n"
               f"n_fake={len(list_images(f'{R}/testA'))}/"
               f"{len(list_images(f'{R}/testB'))}  n_ref={len(both)}\n"
               f"self_A={r['self_T1']:.4f}\nself_B={r['self_FA']:.4f}\n"
               f"fid_A2B={f_ab:.2f}\nfid_B2A={f_ba:.2f}\n"
               f"fid_path_vs_AuB=" + " ".join(f"{v:.2f}" for v in path_fid) + "\n"
               f"shuffle_delta={r['shuf_delta']:.4f}\n"
               f"flow_work={r['flow_work']:.4f}\nlatent_gap={r['latent_gap']:.4f}\n")
        open(os.path.join(RESULTS, "final_eval.txt"), "w").write(txt)
        print(f"\n[FlowCycle/{a.variant}] FINAL  ({R})")
        print(f"  self   A {r['self_T1']:.4f}   B {r['self_FA']:.4f}")
        print(f"  FID    A→B {f_ab:.2f}   B→A {f_ba:.2f}")
        print(f"  FID path vs A∪B: " + "  ".join(f"{v:.1f}" for v in path_fid))
        print(f"  u-shuffle Δoutput {r['shuf_delta']:.4f} "
              f"(0 => the latent is decorative)")
        print(f"  flow_work {r['flow_work']:.4f}   latent_gap {r['latent_gap']:.4f}")
        print("Out:", RESULTS, flush=True)
        log.close()
        return

    txt = (f"variant={a.variant}\n"
           f"self_T1={r['self_T1']:.4f}\nself_FA={r['self_FA']:.4f}\n"
           f"ssim_T1toFA={r['T1toFA']:.4f}\nssim_FAtoT1={r['FAtoT1']:.4f}\n"
           f"shuffled_T1toFA={r['shuf_FA']:.4f}\nshuffled_FAtoT1={r['shuf_T1']:.4f}\n"
           f"shuffle_floor_FA={r['ref_FA']:.4f}\nshuffle_floor_T1={r['ref_T1']:.4f}\n"
           f"flow_work={r['flow_work']:.4f}\nlatent_gap={r['latent_gap']:.4f}\n")
    open(os.path.join(RESULTS, "final_eval.txt"), "w").write(txt)
    print(f"\n[FlowCycle/{a.variant}] FINAL")
    print(f"  self   T1 {r['self_T1']:.4f}   FA {r['self_FA']:.4f}")
    print(f"  cross  T1→FA {r['T1toFA']:.4f}{'✓' if r['T1toFA']>CFM_T1FA else '✗'}   "
          f"FA→T1 {r['FAtoT1']:.4f}{'✓' if r['FAtoT1']>CFM_FAT1 else '✗'}  (vs CFM {CFM_T1FA}/{CFM_FAT1})")
    print(f"  u-shuffle  T1→FA {r['shuf_FA']:.4f} (Δ {r['T1toFA']-r['shuf_FA']:+.4f})   "
          f"FA→T1 {r['shuf_T1']:.4f} (Δ {r['FAtoT1']-r['shuf_T1']:+.4f})")
    print(f"  shuffle floor (two different real slices): FA {r['ref_FA']:.4f}  T1 {r['ref_T1']:.4f}"
          f"   -> latent is load-bearing iff shuffled ≈ floor")
    print(f"  flow_work {r['flow_work']:.4f}   latent_gap {r['latent_gap']:.4f}")
    print("Out:", RESULTS, flush=True)
    log.close()


if __name__ == "__main__":
    main()
