#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired ADNI T1 ↔ FA axial-slice dataset (MNI 2mm, same physical space).

Source files:
    $ADNI_T1_DIR/<subj>_T1_mni2.nii.gz       (91,109,91)
    $ADNI_FA_DIR/<subj>_FA_mni2.nii.gz       (91,109,91)
Produced by data/preprocess/ ; cached to data/cache/paired_112.pt by
data/build_cache.py.

Slices: axial z in [0.35, 0.65] of 91  ->  z = 32..58 (27 slices/subject)
Pad to 112x112 (image_size divisible by 4 for the current encoder).
Per-slice normalization to [0,1] (clip at 99th pct).
Train/test split is BY SUBJECT to avoid leakage.
"""
import os, csv, glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

# Where the MNI-registered volumes live.  ADNI is not redistributable, so these
# point at your own copy; override with the environment variables.
_D = os.path.dirname(os.path.abspath(__file__))
B    = os.environ.get("ADNI_ROOT", "datasets/ADNI")
T1D  = os.environ.get("ADNI_T1_DIR", f"{B}/registrated_T1_sy")
FAD  = os.environ.get("ADNI_FA_DIR", f"{B}/registrated_DTI_2mm_sy")
CACHE      = os.environ.get("ADNI_CACHE",  os.path.join(B, "paired_112.pt"))
LABELS_CSV = os.environ.get("ADNI_LABELS", os.path.join(B, "labels.csv"))

# label-scheme -> int mapping (training is unsupervised; labels are for eval only).
# Any subject whose label is NOT in this map will get -1 and is filtered out
# (e.g. the 1 AD subject under label_4).
LABEL_MAPS = {
    "label_2": {"healthy": 0, "impaired": 1},
    "label_3": {"CN": 0, "MCI": 1, "AD": 2},
    "label_4": {"CN": 0, "EMCI": 1, "MCI": 2, "LMCI": 3},   # AD ("DROP") -> -1
    "label_6": {"CN": 0, "SMC": 1, "EMCI": 2, "MCI": 3, "LMCI": 4, "AD": 5},
}

Z_LO, Z_HI = 32, 59           # inclusive on lo, exclusive on hi  -> 27 slices
IMG = 112                      # padded square size

def pad_to_square(s: np.ndarray, size: int = IMG) -> np.ndarray:
    """Center-pad a 2D slice to size x size with zeros."""
    h, w = s.shape
    out = np.zeros((size, size), dtype=np.float32)
    yo = (size - h) // 2; xo = (size - w) // 2
    out[yo:yo + h, xo:xo + w] = s
    return out

def norm01(s: np.ndarray) -> np.ndarray:
    if not np.any(s > 0): return s
    hi = np.percentile(s[s > 0], 99)
    if hi <= 0: return s
    return np.clip(s / hi, 0, 1).astype(np.float32)

def build_cache(force: bool = False) -> str:
    if os.path.exists(CACHE) and not force:
        return CACHE
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    subjects = sorted({r["subject"] for r in csv.DictReader(open(f"{T1D}/manifest.csv"))
                       if r["status"] == "ok"})
    T1_list, FA_list, sidx, zidx = [], [], [], []
    print(f"Building cache: {len(subjects)} subjects, z={Z_LO}..{Z_HI-1}, pad to {IMG}x{IMG}")
    for i, s in enumerate(subjects):
        t1 = nib.load(f"{T1D}/{s}_T1_mni2.nii.gz").get_fdata().astype(np.float32)
        fa = nib.load(f"{FAD}/{s}_FA_mni2.nii.gz").get_fdata().astype(np.float32)
        for z in range(Z_LO, Z_HI):
            t = pad_to_square(norm01(np.rot90(t1[:, :, z])))   # rot for visual axial
            f = pad_to_square(norm01(np.rot90(fa[:, :, z])))
            T1_list.append(t); FA_list.append(f); sidx.append(i); zidx.append(z)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(subjects)}")
    T1_arr = torch.from_numpy(np.stack(T1_list))[:, None]      # [N,1,H,W]
    FA_arr = torch.from_numpy(np.stack(FA_list))[:, None]
    sidx = torch.tensor(sidx, dtype=torch.long)
    zidx = torch.tensor(zidx, dtype=torch.long)
    torch.save(dict(T1=T1_arr, FA=FA_arr, subj_idx=sidx, z_idx=zidx,
                    subjects=subjects), CACHE)
    print(f"saved -> {CACHE}  T1 {tuple(T1_arr.shape)}  FA {tuple(FA_arr.shape)}")
    return CACHE


def load_subject_labels(scheme: str = "label_2"):
    """Return dict subject_id -> int label, using LABEL_MAPS[scheme]."""
    assert scheme in LABEL_MAPS
    m = LABEL_MAPS[scheme]
    out = {}
    for r in csv.DictReader(open(LABELS_CSV)):
        out[r["subject"]] = m.get(r[scheme], -1)
    return out


class PairedADNISliceDataset(Dataset):
    """Returns (T1, FA, label).

    Sensitivity analysis: `fa_noise` (default: env ADNI_FA_NOISE, else 0) adds Gaussian
    noise of that std to the FA inside the brain mask, clipped to [0,1]. The draw is
    fixed per slice -- seeded by its cache index and `noise_seed` (env
    ADNI_FA_NOISE_SEED, else 0) -- so, like one acquisition, every epoch, split and
    evaluation sees the same noisy FA. A different noise_seed gives an independent
    re-acquisition of the same slice."""
    def __init__(self, indices=None, label_scheme: str = "label_2", fa_noise=None, noise_seed=None):
        d = torch.load(CACHE, map_location="cpu")
        self.T1, self.FA = d["T1"], d["FA"]
        self.subj_idx, self.z_idx = d["subj_idx"], d["z_idx"]
        self.subjects = d["subjects"]
        self.indices = indices if indices is not None else torch.arange(len(self.T1))
        subj2lbl = load_subject_labels(label_scheme)
        self.labels = torch.tensor([subj2lbl.get(self.subjects[int(s)], -1)
                                    for s in self.subj_idx], dtype=torch.long)
        self.fa_noise = float(os.environ.get("ADNI_FA_NOISE", 0)) if fa_noise is None else float(fa_noise)
        self.noise_seed = int(os.environ.get("ADNI_FA_NOISE_SEED", 0)) if noise_seed is None else int(noise_seed)

    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        k = int(self.indices[i])
        t1, fa = self.T1[k], self.FA[k]
        if self.fa_noise > 0:
            g = torch.Generator().manual_seed(self.noise_seed * 1_000_003 + k)
            brain = ((t1 > 0) | (fa > 0)).float()
            fa = (fa + torch.randn(fa.shape, generator=g) * self.fa_noise * brain).clamp(0, 1)
        return t1, fa, int(self.labels[k])


class SingleModalSliceDataset(Dataset):
    """Returns (img, label) for a single modality ('T1' or 'FA')."""
    def __init__(self, modality: str, indices=None, label_scheme: str = "label_2"):
        assert modality in ("T1", "FA")
        d = torch.load(CACHE, map_location="cpu")
        self.X = d["T1"] if modality == "T1" else d["FA"]
        self.subj_idx, self.z_idx = d["subj_idx"], d["z_idx"]
        self.subjects = d["subjects"]
        self.indices = indices if indices is not None else torch.arange(len(self.X))
        subj2lbl = load_subject_labels(label_scheme)
        self.labels = torch.tensor([subj2lbl.get(self.subjects[int(s)], -1)
                                    for s in self.subj_idx], dtype=torch.long)
        self.modality = modality

    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        k = int(self.indices[i])
        return self.X[k], int(self.labels[k])


def subject_level_split(seed: int = 42, test_frac: float = 0.20,
                         label_scheme: str = "label_2",
                         z_lo=None, z_hi=None):
    """Return (train_indices, test_indices) split BY SUBJECT.

    Subjects whose label is invalid under `label_scheme` (mapped to -1) are
    filtered out from BOTH splits.

    `z_lo`/`z_hi` restrict the axial band (inclusive).  Passing 40/49 gives the
    "middle 10" testbed used throughout the paper; see data/build_cache.py
    --report for why.  If left as None they fall back to the ADNI_Z_LO /
    ADNI_Z_HI environment variables, and then to the full z = 32..58 range.
    """
    d = torch.load(CACHE, map_location="cpu")
    subj_idx = d["subj_idx"].numpy()
    subjects = d["subjects"]
    subj2lbl = load_subject_labels(label_scheme)
    valid_subj = [i for i, s in enumerate(subjects) if subj2lbl.get(s, -1) >= 0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(valid_subj))
    n_test = int(round(test_frac * len(valid_subj)))
    test_subj = {valid_subj[i] for i in perm[:n_test].tolist()}
    train_subj = {valid_subj[i] for i in perm[n_test:].tolist()}
    train_idx = torch.from_numpy(np.where(np.isin(subj_idx, list(train_subj)))[0])
    test_idx  = torch.from_numpy(np.where(np.isin(subj_idx, list(test_subj)))[0])
    # Optional axial-slice restriction (env ADNI_Z_LO/ADNI_Z_HI, inclusive).
    # z=40..49 ("middle 10") removes slice-position as the dominant factor —
    # across the full z=32..58 range it swamps individual/pathology variation.
    zlo = int(os.environ.get("ADNI_Z_LO", 0))   if z_lo is None else int(z_lo)
    zhi = int(os.environ.get("ADNI_Z_HI", 999)) if z_hi is None else int(z_hi)
    if (zlo, zhi) != (0, 999):
        zid = d["z_idx"].numpy()
        keep = lambda t: torch.from_numpy(
            np.array([i for i in t.numpy() if zlo <= zid[i] <= zhi], dtype=np.int64))
        train_idx, test_idx = keep(train_idx), keep(test_idx)
        print(f"  [z-restrict {zlo}..{zhi}] -> train {len(train_idx)} / test {len(test_idx)} slices")
    print(f"  split[{label_scheme}]: train={len(train_subj)} subj / {len(train_idx)} slices  "
          f"test={len(test_subj)} subj / {len(test_idx)} slices  "
          f"(dropped {len(subjects) - len(valid_subj)} subj with invalid label)")
    return train_idx, test_idx
