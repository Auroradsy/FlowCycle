#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build labels.csv for our 214 ADNI subjects from kunzhao's all_infomation_v2.json.

Source: /data_new3/nfs_share/kunzhao/TinyLLaVA_Factory-main/dataset/genetic_image/all_infomation_v2.json
Each entry has subject_id, scan_id, label ("The subject is CN" etc.), and a long gene string.
Subject-level label = majority vote across that subject's sessions.

Output: adni_pilot/labels.csv  with columns:
    subject, n_sessions, label_6, label_3, label_2
"""
import json, csv, collections, os

SRC = os.environ["ADNI_LABEL_SOURCE"]
OUR = os.path.join(os.environ["ADNI_RAW_ROOT"], "registrated_T1_sy", "manifest.csv")
OUT = os.environ.get("ADNI_LABELS", "datasets/ADNI/labels.csv")

LABEL_RE = "The subject is "
MAP_6 = {"CN":"CN","SMC":"SMC","EMCI":"EMCI","MCI":"MCI","LMCI":"LMCI","AD":"AD"}
# 4-class: SMC merged into CN (clinically near-normal); drop AD (n=1)
MAP_4 = {"CN":"CN","SMC":"CN","EMCI":"EMCI","MCI":"MCI","LMCI":"LMCI","AD":"DROP"}
MAP_3 = {"CN":"CN","SMC":"CN","EMCI":"MCI","MCI":"MCI","LMCI":"MCI","AD":"AD"}
MAP_2 = {"CN":"healthy","SMC":"healthy","EMCI":"impaired","MCI":"impaired",
         "LMCI":"impaired","AD":"impaired"}

def strip_label(s):  # "The subject is CN" -> "CN"
    return s.replace(LABEL_RE, "").strip()

# our 214 subjects
ours = sorted({r["subject"] for r in csv.DictReader(open(OUR)) if r["status"] == "ok"})
print(f"our subjects: {len(ours)}")

# load source
src = json.load(open(SRC))
print(f"source entries: {len(src)}")

# group by subject
per_subj = collections.defaultdict(list)
for x in src:
    if x["subject_id"] in set(ours):
        per_subj[x["subject_id"]].append(strip_label(x["label"]))

# majority vote per subject (ties broken by counter order)
rows = []
for s in ours:
    labels = per_subj.get(s, [])
    if not labels:
        rows.append(dict(subject=s, n_sessions=0,
                         label_6="UNKNOWN", label_3="UNKNOWN", label_2="UNKNOWN"))
        continue
    ct = collections.Counter(labels); top = ct.most_common(1)[0][0]
    rows.append(dict(subject=s, n_sessions=len(labels),
                     label_6=MAP_6[top], label_4=MAP_4[top],
                     label_3=MAP_3[top], label_2=MAP_2[top]))

# write
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["subject","n_sessions",
                                       "label_6","label_4","label_3","label_2"])
    w.writeheader(); w.writerows(rows)

# summary
for col in ("label_6","label_4","label_3","label_2"):
    ct = collections.Counter(r[col] for r in rows)
    print(f"\n{col} distribution:")
    for k,v in ct.most_common():
        print(f"  {k:10s} {v}")
print(f"\nsaved -> {OUT}")
