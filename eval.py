#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is the shared latent load-bearing?  — a properly calibrated u-shuffle probe.

The in-training probe rolled the batch by one.  That is wrong here: the test
loader is unshuffled and each subject contributes 10 consecutive axial slices,
so roll-by-1 hands the decoder the code of the SAME subject's neighbouring
slice.  Swapping a nearly identical code proves nothing.  This script permutes
ACROSS SUBJECTS instead.

It also reports the two calibration levels an SSIM number on MNI-registered
slices needs before it can be read at all:

  inter-subject floor   SSIM(x_i, x_j), i and j DIFFERENT subjects
                        — what you score by emitting some other real brain
  template floor        SSIM(mean over test set, x_i)
                        — what you score by emitting the dataset mean, i.e. by
                          ignoring the input entirely

A cross-modal SSIM that does not clear these is not measuring subject-specific
fidelity, no matter how it ranks against other methods.

  python eval.py --tag morph
"""
import os
import sys
import argparse
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data.paired_dataset import (                                        # noqa: E402
    build_cache, PairedADNISliceDataset, subject_level_split, CACHE)
from model import FlowCycle                                              # noqa: E402
from utils.image import to_pm1, to_01, ssim_batch                        # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from server_paths import experiment_root, checkpoint_root
EXPS = experiment_root()
CKPT = os.path.join(EXPS, "checkpoints")


def cross_subject_perm(subj, seed=0):
    """A permutation in which no element keeps its own subject."""
    rng = np.random.default_rng(seed)
    n = len(subj)
    for _ in range(200):
        p = rng.permutation(n)
        if (subj[p] != subj).all():
            return p
    # fall back: fix collisions by rotating within the offending positions
    p = rng.permutation(n)
    bad = np.where(subj[p] == subj)[0]
    for i in bad:
        for j in range(n):
            if subj[p[j]] != subj[i] and subj[p[i]] != subj[j]:
                p[i], p[j] = p[j], p[i]
                break
    return p


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="morph")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--z_lo", type=int, default=40)
    ap.add_argument("--z_hi", type=int, default=49)
    ap.add_argument("--label_scheme", default="label_4")
    a = ap.parse_args()

    ck = torch.load(os.path.join(CKPT, a.tag, "model.pth"), map_location=DEV)
    cfg = ck["args"]
    m = FlowCycle(cfg["ngf"], cfg["n_blocks"], cfg["n_flow"], cfg["flow_hidden"],
                  bool(cfg["pre_relu"])).to(DEV)
    m.load_state_dict(ck["model"]); m.eval()

    build_cache()
    _, te = subject_level_split(42, 0.20, a.label_scheme, a.z_lo, a.z_hi)
    ds = PairedADNISliceDataset(te, a.label_scheme)
    subj = torch.load(CACHE, map_location="cpu")["subj_idx"].numpy()[te.numpy()]
    n = len(ds)
    print(f"test: {n} slices / {len(np.unique(subj))} subjects  (model: {a.tag})")

    # --- collect codes and images in one pass ---
    T1, FA, Z, U = [], [], [], []
    for i in range(0, n, 32):
        t1 = torch.stack([ds[j][0] for j in range(i, min(i + 32, n))])
        fa = torch.stack([ds[j][1] for j in range(i, min(i + 32, n))])
        T1.append(t1); FA.append(fa)
        Z.append(m.enc_A(to_pm1(t1.to(DEV))).cpu())
        U.append(m.enc_B(to_pm1(fa.to(DEV))).cpu())
    T1 = torch.cat(T1); FA = torch.cat(FA); Z = torch.cat(Z); U = torch.cat(U)

    perm = cross_subject_perm(subj, a.seed)
    assert (subj[perm] != subj).all(), "permutation kept a same-subject pairing"

    def decode(codes, dec, forward):
        out = []
        for i in range(0, len(codes), 32):
            c = codes[i:i + 32].to(DEV)
            c = m.a_to_b(c) if forward else m.b_to_a(c)
            out.append(to_01(dec(c)).cpu())
        return torch.cat(out)

    gen_FA = decode(Z, m.dec_B, True)                 # T1 -> FA, correct code
    gen_T1 = decode(U, m.dec_A, False)                # FA -> T1, correct code
    shf_FA = decode(Z[perm], m.dec_B, True)           # another SUBJECT's code
    shf_T1 = decode(U[perm], m.dec_A, False)

    mean_FA = FA.mean(0, keepdim=True).expand_as(FA)
    mean_T1 = T1.mean(0, keepdim=True).expand_as(T1)

    def score(pred, gt):
        return ssim_batch(pred, gt)

    rows = [
        ("T1→FA  (correct code)", score(gen_FA, FA), "FA→T1  (correct code)", score(gen_T1, T1)),
        ("T1→FA  (other subject's code)", score(shf_FA, FA),
         "FA→T1  (other subject's code)", score(shf_T1, T1)),
        ("inter-subject floor", score(FA[perm], FA), "inter-subject floor", score(T1[perm], T1)),
        ("template floor (dataset mean)", score(mean_FA, FA),
         "template floor (dataset mean)", score(mean_T1, T1)),
    ]
    print(f"\n{'':34s}{'FA target':>11s}   {'T1 target':>11s}")
    for la, va, _lb, vb in rows:
        print(f"{la:34s}{va:11.4f}   {vb:11.4f}")

    d_fa = rows[0][1] - rows[1][1]; d_t1 = rows[0][3] - rows[1][3]
    print(f"\nu-shuffle drop:   FA {d_fa:+.4f}   T1 {d_t1:+.4f}"
          f"   (FiLM head of §3: 0.0000 = decorative latent)")
    print(f"headroom over template floor:   FA {rows[0][1]-rows[3][1]:+.4f}"
          f"   T1 {rows[0][3]-rows[3][3]:+.4f}")
    out = os.path.join(EXPS, "probe_eval.txt")
    with open(out, "w") as f:
        for la, va, _lb, vb in rows:
            f.write(f"{la}\tFA={va:.4f}\tT1={vb:.4f}\n")
        f.write(f"u_shuffle_drop\tFA={d_fa:.4f}\tT1={d_t1:.4f}\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
