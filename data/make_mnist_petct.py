#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render MNIST into a CT/PET pair and write it in CycleGAN folder layout.

Why this dataset exists
-----------------------
horse2zebra answered "does FlowCycle survive without pairing", but it could
not answer "is the translation any GOOD", because unpaired data admits no SSIM
against ground truth.  All we had was FID (set-level, needs calibration) and
cycle SSIM (which an identity map wins).  That ambiguity is what left the
endpoint claim unresolved.

MNIST-PET/CT removes it.  The PET modality is DERIVED from the MNIST image, so
a ground-truth counterpart exists for every sample — but only if we choose to
expose it.  So the split is deliberately asymmetric:

    train/  DISJOINT indices between A and B.  No image's own PET counterpart
            appears in the other domain, so training is genuinely unpaired and
            the horse2zebra protocol carries over unchanged.
    test/   The SAME indices in both, written in matching filename order, so
            testA/000042.png and testB/000042.png are the same digit.

Training therefore never sees a pair, while evaluation gets exact ones.  That
is the thing horse2zebra could not provide.

The modalities
--------------
A = CT  = the ORIGINAL MNIST digit, grayscale replicated to RGB.  No styling.
B = PET = per-class hotspot boost -> PET intensity pipeline -> green tint.

The asymmetry is the point.  PET carries a class-conditional functional signal
(the hotspots) that simply is not present in CT, so CT->PET must INVENT it and
can only ever be right distributionally, while PET->CT merely has to remove it
and is well-posed.  Expect the two directions to behave differently; on
horse2zebra that asymmetry was present but unexplained.

Sizing: the hotspot sigmas and the PET blur are tuned for 28x28, so the whole
pipeline runs at native resolution and the result is upscaled afterwards.
Doing it the other way round would shrink every blob by the scale factor.

    python -m data.make_mnist_petct                       # defaults
    python -m data.make_mnist_petct --hotspot_weight 1.5  # heavy preset
"""
import argparse
import os
import struct

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# Copied from __outdated_files/synth_datas/hotspot_config.py rather than
# imported: that tree is staged for deletion, and this table is the only part
# of it this dataset depends on.  Entries are (row_rel, col_rel, sigma) in
# stroke-bbox-relative coordinates.
REL_REGIONS = {
    0: [(0.10, 0.50, 2.5), (0.50, 0.10, 2.5), (0.50, 0.90, 2.5), (0.90, 0.50, 2.5)],
    1: [(0.00, 0.50, 2.5), (1.00, 0.50, 2.5)],
    2: [(0.10, 0.10, 2.5), (0.10, 0.90, 2.5), (0.80, 0.20, 2.5), (0.75, 0.90, 2.5)],
    3: [(0.15, 0.80, 2.5), (0.50, 0.20, 2.5), (0.85, 0.50, 2.5)],
    4: [(0.15, 0.15, 2.5), (0.10, 0.80, 2.5), (0.50, 0.90, 2.5), (0.90, 0.90, 2.5)],
    5: [(0.25, 0.25, 2.5), (0.10, 0.85, 2.5), (0.85, 0.85, 2.5)],
    6: [(0.10, 0.50, 2.5), (0.90, 0.10, 2.5), (0.50, 0.75, 2.5)],
    7: [(0.10, 0.15, 2.5), (0.10, 0.90, 2.5), (0.95, 0.50, 2.5)],
    8: [(0.50, 0.50, 2.5), (0.10, 0.85, 2.5), (0.90, 0.15, 2.5)],
    9: [(0.15, 0.85, 2.5), (0.50, 0.70, 2.5), (0.85, 0.85, 2.5)],
}
STROKE_THRESHOLD = 0.1
SNAP_MAX_DIST = 4

_DEFAULT_MNIST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "__outdated_files", "data", "MNIST", "raw")


# --------------------------------------------------------------------------
# idx readers — avoids torchvision's download path and its version churn
# --------------------------------------------------------------------------
def read_idx_images(p):
    with open(p, "rb") as f:
        magic, n, r, c = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"{p}: bad magic {magic}"
        return np.frombuffer(f.read(n * r * c), np.uint8).reshape(n, r, c)


def read_idx_labels(p):
    with open(p, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"{p}: bad magic {magic}"
        return np.frombuffer(f.read(n), np.uint8)


# --------------------------------------------------------------------------
# hotspots (verbatim behaviour from synth_datas/mnist_petct.py)
# --------------------------------------------------------------------------
def _snap_to_stroke(r, c, mask, max_dist=SNAP_MAX_DIST):
    rows, cols = np.where(mask)
    if not len(rows):
        return None
    d2 = (rows - r) ** 2 + (cols - c) ** 2
    i = int(np.argmin(d2))
    return None if d2[i] > max_dist ** 2 else (int(rows[i]), int(cols[i]))


def _boost_hotspots(x, label, weight):
    """Multiplicative, stroke-gated boost.  Background is left untouched so the
    hotspot reads as local tracer uptake rather than a blob pasted on top."""
    if weight <= 0:
        return x
    stroke = x > STROKE_THRESHOLD
    rows, cols = np.where(stroke)
    if not len(rows):
        return x
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    h, w = max(r1 - r0, 1), max(c1 - c0, 1)
    rr, cc = np.meshgrid(np.arange(x.shape[0]), np.arange(x.shape[1]), indexing="ij")
    blob = np.zeros_like(x)
    for rh, rc, sigma in REL_REGIONS.get(int(label), []):
        sp = _snap_to_stroke(int(r0 + rh * h), int(c0 + rc * w), stroke)
        if sp is not None:
            blob = np.maximum(blob, np.exp(-((rr - sp[0]) ** 2 + (cc - sp[1]) ** 2)
                                           / (2 * sigma ** 2)))
    return np.clip(x * (1 + blob * stroke.astype(np.float32) * weight), 0, 1)


def _pet_intensity(x, rng, noise=0.05):
    """PSF blur + SUV-style display curve + acquisition noise.  The +0.12 lift
    is what makes the PET background non-black, which is most of the visible
    domain gap against raw MNIST."""
    x = gaussian_filter(x, sigma=1.2)
    x = np.clip(x, 0, 1) ** 2.2
    x = np.clip(x * 1.15 + 0.12, 0, 1)
    return np.clip(x + rng.normal(0, noise, x.shape), 0, 1).astype(np.float32)


def render_ct(x28, rng):
    """Original MNIST, grayscale replicated to RGB.  Deliberately unstyled."""
    return np.stack([x28] * 3, -1)


def render_pet(x28, label, rng, weight, noise=0.05):
    g = _pet_intensity(_boost_hotspots(x28, label, weight), rng, noise)
    rgb = np.zeros((*g.shape, 3), np.float32)
    rgb[..., 1] = g                      # pure-green tint, R=B=0
    return rgb


def save(arr_hw3, path, size):
    im = Image.fromarray((np.clip(arr_hw3, 0, 1) * 255).astype(np.uint8))
    if size != im.size[0]:
        im = im.resize((size, size), Image.BICUBIC)
    im.save(path)


def balanced_indices(labels, per_class, offset=0):
    """`offset` classes apart lets A and B draw DISJOINT index ranges."""
    out = []
    for d in range(10):
        idx = np.where(labels == d)[0]
        take = idx[offset:offset + per_class]
        if len(take) < per_class:
            raise SystemExit(f"digit {d}: need {per_class} from offset {offset}, "
                             f"have {len(take)} (class size {len(idx)})")
        out.append(take)
    return np.sort(np.concatenate(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist_raw", default=os.path.normpath(_DEFAULT_MNIST))
    ap.add_argument("--out", default="datasets/mnist_petct")
    ap.add_argument("--size", type=int, default=64,
                    help="output resolution; the pipeline always runs at 28")
    ap.add_argument("--n_train", type=int, default=12000,
                    help="per domain; A and B use disjoint indices")
    ap.add_argument("--n_test", type=int, default=5000,
                    help="per domain; A and B use the SAME indices (paired)")
    ap.add_argument("--hotspot_weight", type=float, default=0.8)
    ap.add_argument("--pet_noise", type=float, default=0.05,
                    help="std of the additive Gaussian acquisition noise on PET "
                         "(after the display curve, in [0,1] intensity units)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--paired_train", action="store_true",
                    help="give trainB the SAME indices as trainA, so every CT "
                         "has its own PET counterpart in the training set.  "
                         "Enables the ADNI protocol (L_cross, SSIM-vs-truth) on "
                         "this data.  The default disjoint split exists to make "
                         "'unpaired' an honest description; this flag is the "
                         "controlled opposite of it.")
    ap.add_argument("--aux_nohot", action="store_true",
                    help="also write testB_nohot: the same PET images with the "
                         "hotspot boost off and the SAME noise draws, so "
                         "testB - testB_nohot isolates the class-conditional "
                         "signal.  Diagnostic only; never trained on.")
    a = ap.parse_args()

    if a.n_train % 10 or a.n_test % 10:
        raise SystemExit("--n_train / --n_test must be multiples of 10 (balanced)")

    tr_x = read_idx_images(os.path.join(a.mnist_raw, "train-images-idx3-ubyte"))
    tr_y = read_idx_labels(os.path.join(a.mnist_raw, "train-labels-idx1-ubyte"))
    te_x = read_idx_images(os.path.join(a.mnist_raw, "t10k-images-idx3-ubyte"))
    te_y = read_idx_labels(os.path.join(a.mnist_raw, "t10k-labels-idx1-ubyte"))
    print(f"MNIST: train {tr_x.shape}, test {te_x.shape}")

    pc_tr, pc_te = a.n_train // 10, a.n_test // 10
    # train: B starts where A ends -> no image's own counterpart is across the
    # domain gap, which is what makes "unpaired" an honest description.
    idx = {
        ("train", "A"): balanced_indices(tr_y, pc_tr, 0),
        ("train", "B"): balanced_indices(tr_y, pc_tr,
                                        0 if a.paired_train else pc_tr),
        ("test", "A"): balanced_indices(te_y, pc_te, 0),
        ("test", "B"): balanced_indices(te_y, pc_te, 0),   # SAME -> paired eval
    }

    for split, dom in idx:
        os.makedirs(os.path.join(a.out, split + dom), exist_ok=True)

    for (split, dom), ii in idx.items():
        X, Y = (tr_x, tr_y) if split == "train" else (te_x, te_y)
        d = os.path.join(a.out, split + dom)
        rng = np.random.RandomState(a.seed + (0 if dom == "A" else 1)
                                    + (0 if split == "train" else 100))
        for k, i in enumerate(ii):
            x = X[i].astype(np.float32) / 255.0
            img = (render_ct(x, rng) if dom == "A"
                   else render_pet(x, Y[i], rng, a.hotspot_weight, a.pet_noise))
            # filename carries the source index, so testA/testB line up by name
            save(img, os.path.join(d, f"{i:06d}_d{Y[i]}.png"), a.size)
        print(f"{split}{dom}: {len(ii)} -> {d}")

    if a.aux_nohot:
        d = os.path.join(a.out, "testB_nohot")
        os.makedirs(d, exist_ok=True)
        # Same seed and same iteration order as the testB pass above, so the
        # rng draws line up call-for-call and the only difference is the boost.
        rng = np.random.RandomState(a.seed + 1 + 100)
        for i in idx[("test", "B")]:
            x = te_x[i].astype(np.float32) / 255.0
            save(render_pet(x, te_y[i], rng, 0.0, a.pet_noise),
                 os.path.join(d, f"{i:06d}_d{te_y[i]}.png"), a.size)
        print(f"testB_nohot: {len(idx[('test', 'B')])} -> {d}")

    ov = set(idx[("train", "A")]) & set(idx[("train", "B")])
    want = len(idx[("train", "A")]) if a.paired_train else 0
    print(f"\ntrain A/B index overlap: {len(ov)} "
          f"(must be {want} for {'paired' if a.paired_train else 'unpaired'})")
    assert len(ov) == want, "train split does not match the requested pairing"
    ta = sorted(os.listdir(os.path.join(a.out, "testA")))
    tb = sorted(os.listdir(os.path.join(a.out, "testB")))
    print(f"test filenames aligned: {ta == tb} ({len(ta)} pairs)")


if __name__ == "__main__":
    main()
