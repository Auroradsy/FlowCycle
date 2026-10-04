#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired scoring for MNIST-PET/CT — the metric horse2zebra could not provide.

Training is unpaired (trainA and trainB draw disjoint MNIST indices), but the
PET modality is derived from the CT one, so the TEST split can expose exact
pairs.  That closes the gap the horse2zebra results left open: there, quality
was FID alone, and the only SSIM available was cycle SSIM, which an identity
map wins outright.  Here every translation is scored against its own ground
truth.

Three families of number, each with its own floor and ceiling so a raw value
cannot be over-read (the same discipline the h2z FID calibration needed):

  SSIM / PSNR vs GT   the headline.  Ceiling is 1.0; the meaningful floor is
                      `copy`, the score for emitting the input unchanged.  An
                      arm that does not beat `copy` has not translated.

  cycle SSIM          A -> B -> A.  Carried over from h2z for continuity.  NOT
                      a quality metric: an identity map scores 1.0.

  hotspot contrast    the interesting one.  PET carries per-class hotspots that
                      simply are not in CT, so A->B must INVENT them.  Using
                      testB_nohot (identical noise draws, boost off):

                          M       = pixels the boost materially raised
                          contrast= mean(X[M]) / mean(X[stroke \ M])

                      real PET gives the ceiling, nohot PET the floor (~1.0 by
                      construction).  A model that ignores class identity lands
                      on the floor no matter how good its SSIM is.

    python -m utils.eval_mnist                 # host + every finished mnist_* arm
    python -m utils.eval_mnist --tags mnist_base
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model import FlowCycle                                               # noqa: E402
from model.backbone import ResnetGenerator                                # noqa: E402
from utils.image import to_pm1, to_01                                     # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from server_paths import experiment_root, checkpoint_root
EXPS = experiment_root()
CKPT = os.path.join(EXPS, "checkpoints")
OUT = os.path.join(EXPS, "snapshot_results", "mnist_petct")

HOT_TAU = 0.04        # boost that counts as "materially raised", in [0,1]
STROKE_TAU = 0.10     # matches STROKE_THRESHOLD in data/make_mnist_petct.py


class PairedTest(Dataset):
    """testA / testB / testB_nohot, aligned by filename.

    The alignment is asserted rather than assumed: it is the single property
    the whole paired evaluation rests on, and a silent mismatch would produce
    plausible-looking numbers that mean nothing.
    """

    def __init__(self, root, size=64):
        self.root, self.size = root, size
        ls = lambda d: sorted(f for f in os.listdir(os.path.join(root, d))
                              if f.endswith(".png"))
        self.fa, fb = ls("testA"), ls("testB")
        if self.fa != fb:
            raise SystemExit("testA / testB filenames do not align — regenerate "
                             "with data.make_mnist_petct; paired scoring is invalid")
        nh = os.path.join(root, "testB_nohot")
        self.has_nohot = os.path.isdir(nh) and sorted(
            f for f in os.listdir(nh) if f.endswith(".png")) == self.fa
        self.labels = [int(f.split("_d")[1][0]) for f in self.fa]

    def __len__(self):
        return len(self.fa)

    def _load(self, sub, f):
        im = Image.open(os.path.join(self.root, sub, f)).convert("RGB")
        if im.size[0] != self.size:
            im = im.resize((self.size, self.size), Image.BICUBIC)
        return torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)

    def __getitem__(self, i):
        f = self.fa[i]
        nh = self._load("testB_nohot", f) if self.has_nohot else torch.zeros(1)
        return self._load("testA", f), self._load("testB", f), nh, self.labels[i]


def ssim_psnr(pred, gt):
    from skimage.metrics import structural_similarity as ssim
    p = to_01(pred).cpu().numpy(); g = to_01(gt).cpu().numpy()
    s, q = [], []
    for i in range(p.shape[0]):
        a, b = g[i].transpose(1, 2, 0), p[i].transpose(1, 2, 0)
        s.append(ssim(a, b, data_range=1.0, channel_axis=2))
        mse = float(np.mean((a - b) ** 2))
        q.append(10 * np.log10(1.0 / max(mse, 1e-12)))
    return s, q


def hotspot_contrast(x, pet_real, pet_nohot, ct):
    """mean(x[boosted]) / mean(x[stroke but not boosted]), on the green channel.

    The masks are ALWAYS derived from the real pair (`pet_real` minus
    `pet_nohot`), never from `x`.  That is what makes the score comparable:
    every model — and both references — is measured on exactly the same pixels,
    and only the intensities inside them are its own.  Deriving the mask from
    `x` would let the floor measure `N - N`, which is identically zero.
    """
    g = to_01(x)[:, 1]
    hot = (to_01(pet_real)[:, 1] - to_01(pet_nohot)[:, 1]) > HOT_TAU
    stroke = to_01(ct).mean(1) > STROKE_TAU
    ctrl = stroke & ~hot
    out = []
    for i in range(g.shape[0]):
        if hot[i].sum() < 4 or ctrl[i].sum() < 4:
            continue                                      # not measurable
        out.append(float(g[i][hot[i]].mean() / (g[i][ctrl[i]].mean() + 1e-8)))
    return out


@torch.no_grad()
def score(fwd, bwd, dl, want_hot=True):
    acc = {k: [] for k in ["ssim_A2B", "psnr_A2B", "ssim_B2A", "psnr_B2A",
                           "cyc_A", "cyc_B", "hot_A2B"]}
    for xa, xb, nh, _ in dl:
        A, B = to_pm1(xa.to(DEV)), to_pm1(xb.to(DEV))
        fb, fa = fwd(A), bwd(B)
        s, q = ssim_psnr(fb, B); acc["ssim_A2B"] += s; acc["psnr_A2B"] += q
        s, q = ssim_psnr(fa, A); acc["ssim_B2A"] += s; acc["psnr_B2A"] += q
        acc["cyc_A"] += ssim_psnr(bwd(fb), A)[0]
        acc["cyc_B"] += ssim_psnr(fwd(fa), B)[0]
        if want_hot:
            N = to_pm1(nh.to(DEV))
            acc["hot_A2B"] += hotspot_contrast(fb, B, N, A)
    return {k: (float(np.mean(v)) if v else None) for k, v in acc.items()}


@torch.no_grad()
def references(dl):
    """copy (emit the input), and the hotspot floor/ceiling from the real data."""
    acc = {k: [] for k in ["ssim_A2B", "psnr_A2B", "ssim_B2A", "psnr_B2A"]}
    hot_real, hot_floor = [], []
    for xa, xb, nh, _ in dl:
        A, B, N = to_pm1(xa.to(DEV)), to_pm1(xb.to(DEV)), to_pm1(nh.to(DEV))
        s, q = ssim_psnr(A, B); acc["ssim_A2B"] += s; acc["psnr_A2B"] += q
        s, q = ssim_psnr(B, A); acc["ssim_B2A"] += s; acc["psnr_B2A"] += q
        hot_real += hotspot_contrast(B, B, N, A)     # ceiling: the real PET
        hot_floor += hotspot_contrast(N, B, N, A)    # floor: boost off, same mask
    r = {k: float(np.mean(v)) for k, v in acc.items()}
    r["hot_A2B"] = float(np.mean(hot_real))
    return r, float(np.mean(hot_floor))


def _host_n_blocks(ck, default):
    """n_blocks lives in ck["args"], not at the top level.  ck.get("n_blocks", D)
    therefore always returned D silently -- correct only when D happened to
    match the run.  Read the real value and fall back only if it is absent."""
    a = ck.get("args") or {}
    return int(a.get("n_blocks", ck.get("n_blocks", default)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/mnist_petct_paired")
    ap.add_argument("--warm", default=os.path.join(CKPT, "mnist_host", "last.pth"))
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="0 = all test pairs")
    a = ap.parse_args()

    ds = PairedTest(a.root)
    if a.limit:
        ds.fa = ds.fa[:a.limit]; ds.labels = ds.labels[:a.limit]
    print(f"{len(ds)} paired test images  (testB_nohot: {ds.has_nohot})")
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=4)

    res = {}
    ref, hot_floor = references(dl)
    res["_copy (do nothing)"] = {**ref, "hot_A2B": None}
    res["_real PET (ceiling)"] = {"ssim_A2B": 1.0, "psnr_A2B": None,
                                  "ssim_B2A": 1.0, "psnr_B2A": None,
                                  "cyc_A": None, "cyc_B": None,
                                  "hot_A2B": ref["hot_A2B"]}
    res["_nohot PET (hot floor)"] = {**{k: None for k in ref},
                                     "hot_A2B": hot_floor}

    if os.path.exists(a.warm):
        ck = torch.load(a.warm, map_location=DEV)
        gs = {}
        for key, name in [("G_T1toFA", "a2b"), ("G_FAtoT1", "b2a")]:
            g = ResnetGenerator(3, 3, 64, _host_n_blocks(ck, 6)).to(DEV).eval()
            g.load_state_dict(ck[key], strict=True)
            gs[name] = g
        res["host"] = score(gs["a2b"], gs["b2a"], dl)
    else:
        print(f"no host at {a.warm} — skipping")

    tags = a.tags if a.tags is not None else sorted(
        t for t in os.listdir(CKPT)
        if t.startswith("mnist_") and t != "mnist_host"
        and os.path.exists(os.path.join(CKPT, t, "model.pth")))
    for t in tags:
        c = torch.load(os.path.join(CKPT, t, "model.pth"), map_location=DEV)
        ar = c["args"]
        m = FlowCycle(ar["ngf"], ar["n_blocks"], ar["n_flow"], ar["flow_hidden"],
                      bool(ar["pre_relu"]), img_ch=ar.get("img_ch", 3)).to(DEV)
        m.load_state_dict(c["model"]); m.eval()
        res[t] = score(m.cross_A2B, m.cross_B2A, dl)
        del m; torch.cuda.empty_cache()

    # Reference rows carry only the columns that are defined for them; fill the
    # rest so the table prints a dash rather than raising.
    KEYS = ["ssim_A2B", "psnr_A2B", "ssim_B2A", "psnr_B2A",
            "cyc_A", "cyc_B", "hot_A2B"]
    res = {k: {c: v.get(c) for c in KEYS} for k, v in res.items()}

    f = lambda v, w=8, p=4: (" " * (w - 1) + "-") if v is None else f"{v:{w}.{p}f}"
    hdr = (f"{'':24s} {'SSIM A→B':>9s} {'PSNR':>7s} {'SSIM B→A':>9s} {'PSNR':>7s} "
           f"| {'cyc A':>7s} {'cyc B':>7s} | {'hotspot':>8s}")
    print("\n" + hdr); print("-" * len(hdr))
    for k, d in res.items():
        print(f"{k:24s} {f(d['ssim_A2B'],9)} {f(d['psnr_A2B'],7,2)} "
              f"{f(d['ssim_B2A'],9)} {f(d['psnr_B2A'],7,2)} | "
              f"{f(d['cyc_A'],7)} {f(d['cyc_B'],7)} | {f(d['hot_A2B'],8,3)}")
    print("\nSSIM/PSNR are against GROUND TRUTH — beat `_copy` or nothing "
          "translated.\nhotspot: floor = `_nohot`, ceiling = `_real PET`.  It is "
          "the only column\n  that asks whether class-conditional signal absent "
          "from CT was invented.")

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "paired_table.json")
    json.dump(res, open(p, "w"), indent=2)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
