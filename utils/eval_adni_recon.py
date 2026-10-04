#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reconstruction metrics (SSIM + PSNR) for the ADNI runs, scored post hoc.

The training script writes SSIM into `final_eval.txt` but not PSNR, and the
host writes both — so the two are not directly comparable as stored.  This
recomputes every number under one protocol, on the same subject-level test
split the runs used, so a paper table can be built from a single source.

Mirrors `utils.eval_mnist`, including the `_copy` row: the score for emitting
the input unchanged.  On MNI-registered slices that floor is high, and a
cross-modal SSIM that does not clear it is not measuring anything modality
-specific.

    python -m utils.eval_adni_recon                 # host + every finished arm
    python -m utils.eval_adni_recon --tags morph
"""
import argparse
import json
import os
import sys
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data import build_cache, PairedADNISliceDataset, subject_level_split      # noqa: E402
from model import FlowCycle                                                    # noqa: E402
from model.backbone import ResnetGenerator                                     # noqa: E402
from utils.image import to_pm1, to_01                                          # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from server_paths import experiment_root, checkpoint_root
EXPS = experiment_root()
CKPT = os.path.join(EXPS, "checkpoints")
OUT = os.path.join(EXPS, "snapshot_results")


def _host_n_blocks(ck, default):
    a = ck.get("args") or {}
    return int(a.get("n_blocks", ck.get("n_blocks", default)))


def ssim_psnr(pred, gt):
    from skimage.metrics import structural_similarity as ssim
    p = to_01(pred).cpu().numpy(); g = to_01(gt).cpu().numpy()
    s, q = [], []
    for i in range(p.shape[0]):
        a, b = g[i, 0], p[i, 0]                      # ADNI is single-channel
        s.append(ssim(a, b, data_range=1.0))
        mse = float(np.mean((a - b) ** 2))
        q.append(10 * np.log10(1.0 / max(mse, 1e-12)))
    return s, q


@torch.no_grad()
def score(fwd, bwd, dl):
    """fwd: T1->FA, bwd: FA->T1."""
    acc = {k: [] for k in ["ssim_T1toFA", "psnr_T1toFA",
                           "ssim_FAtoT1", "psnr_FAtoT1"]}
    for t1, fa, _ in dl:
        T, F = to_pm1(t1.to(DEV)), to_pm1(fa.to(DEV))
        s, q = ssim_psnr(fwd(T), F); acc["ssim_T1toFA"] += s; acc["psnr_T1toFA"] += q
        s, q = ssim_psnr(bwd(F), T); acc["ssim_FAtoT1"] += s; acc["psnr_FAtoT1"] += q
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in acc.items()}


@torch.no_grad()
def copy_floor(dl):
    acc = {k: [] for k in ["ssim_T1toFA", "psnr_T1toFA",
                           "ssim_FAtoT1", "psnr_FAtoT1"]}
    for t1, fa, _ in dl:
        T, F = to_pm1(t1.to(DEV)), to_pm1(fa.to(DEV))
        s, q = ssim_psnr(T, F); acc["ssim_T1toFA"] += s; acc["psnr_T1toFA"] += q
        s, q = ssim_psnr(F, T); acc["ssim_FAtoT1"] += s; acc["psnr_FAtoT1"] += q
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", default=os.path.join(CKPT, "host", "last.pth"))
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label_scheme", default="label_4")
    ap.add_argument("--z_lo", type=int, default=40)
    ap.add_argument("--z_hi", type=int, default=49)
    a = ap.parse_args()

    build_cache()
    _, te_idx = subject_level_split(a.seed, 0.20, a.label_scheme, a.z_lo, a.z_hi)
    ds = PairedADNISliceDataset(te_idx, a.label_scheme)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
    print(f"{len(te_idx)} test slices")

    res = {"_copy (do nothing)": copy_floor(dl)}

    if os.path.exists(a.warm):
        ck = torch.load(a.warm, map_location=DEV)
        nb = _host_n_blocks(ck, 6)
        g = {}
        for key, name in [("G_T1toFA", "fwd"), ("G_FAtoT1", "bwd")]:
            n = ResnetGenerator(1, 1, 64, nb).to(DEV).eval()
            n.load_state_dict(ck[key], strict=True)
            g[name] = n
        res["host"] = score(g["fwd"], g["bwd"], dl)

    skip = {"host"}
    tags = a.tags if a.tags is not None else sorted(
        t for t in os.listdir(CKPT)
        if t not in skip and not t.startswith(("h2z_", "mnist_"))
        and os.path.exists(os.path.join(CKPT, t, "model.pth")))
    for t in tags:
        c = torch.load(os.path.join(CKPT, t, "model.pth"), map_location=DEV)
        ar = c["args"]
        m = FlowCycle(ar["ngf"], ar["n_blocks"], ar["n_flow"], ar["flow_hidden"],
                      bool(ar["pre_relu"]), img_ch=ar.get("img_ch", 1)).to(DEV)
        m.load_state_dict(c["model"]); m.eval()
        res[t] = score(m.cross_A2B, m.cross_B2A, dl)
        del m; torch.cuda.empty_cache()

    hdr = (f"{'':18s} {'SSIM T1→FA':>16s} {'PSNR':>7s} "
           f"{'SSIM FA→T1':>16s} {'PSNR':>7s}")
    print("\n" + hdr); print("-" * len(hdr))
    for k, d in res.items():
        print(f"{k:18s} {d['ssim_T1toFA'][0]:8.4f}±{d['ssim_T1toFA'][1]:.4f} "
              f"{d['psnr_T1toFA'][0]:7.2f} "
              f"{d['ssim_FAtoT1'][0]:8.4f}±{d['ssim_FAtoT1'][1]:.4f} "
              f"{d['psnr_FAtoT1'][0]:7.2f}")

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "adni_recon_table.json")
    json.dump(res, open(p, "w"), indent=2)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
