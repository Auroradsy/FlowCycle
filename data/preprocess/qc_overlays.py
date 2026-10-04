#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Random QC sample from the batch: 6 subjects (incl slowest), FA-edge on T1 overlay."""
import os, glob, csv, random
import numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import feature
from nilearn.image import resample_to_img

B = os.environ["ADNI_RAW_ROOT"]
TDIR = f"{B}/registrated_T1_sy"
MNI2 = os.path.join(os.environ.get("FSLDIR", "/usr/local/fsl"), "data/standard/MNI152_T1_2mm.nii.gz")
OVR = os.environ.get("ADNI_QC_OUT", "exps/adni/qc/overlays")

rows = list(csv.DictReader(open(f"{TDIR}/manifest.csv")))
rows = [r for r in rows if r["status"] == "ok"]
slowest = max(rows, key=lambda r: float(r["seconds"]))
random.seed(0)
sample = [slowest] + random.sample([r for r in rows if r is not slowest], 5)
print("QC subjects:", [(r["subject"], r["seconds"]) for r in sample])

def first_fa(subj):
    for d in sorted(glob.glob(f"{B}/registered_DTI/{subj}_I*")):
        h = glob.glob(f"{d}/*_FA_reg.nii.gz")
        if h: return h[0]
    return None

def norm(x):
    x = x.astype(float)
    if (x>0).any(): lo, hi = np.percentile(x[x>0],[1,99])
    else: lo, hi = 0, 1
    return np.clip((x-lo)/(hi-lo+1e-8), 0, 1)

fig, axes = plt.subplots(len(sample), 3, figsize=(9, 3*len(sample)))
zfracs = (0.40, 0.52, 0.65)
for r, row in enumerate(sample):
    s = row["subject"]
    t1 = nib.load(f"{TDIR}/{s}_T1_mni2.nii.gz").get_fdata()
    fa_path = first_fa(s)
    fa = resample_to_img(fa_path, MNI2, interpolation="continuous").get_fdata()
    for c, zf in enumerate(zfracs):
        z = int(round(zf * t1.shape[2]))
        t1s = norm(np.rot90(t1[:,:,z])); fas = norm(np.rot90(fa[:,:,z]))
        edges = feature.canny(fas, sigma=1.5)
        em = np.ma.masked_where(~edges, edges)
        axes[r,c].imshow(t1s, cmap="gray")
        axes[r,c].imshow(em, cmap="cool", alpha=0.9)
        axes[r,c].set_xticks([]); axes[r,c].set_yticks([])
    axes[r,0].set_ylabel(f"{s}\n{row['seconds']}s", fontsize=9)
for c, zf in enumerate(zfracs): axes[0,c].set_title(f"z={zf}", fontsize=10)
fig.suptitle("Batch QC: FA edges (cyan) on registered T1 — random 6 subjects (incl. slowest)", fontsize=11)
fig.tight_layout(rect=[0,0,1,0.97])
out = f"{OVR}/batch_qc.png"; fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
