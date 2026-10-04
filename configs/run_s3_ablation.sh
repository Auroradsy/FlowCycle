#!/usr/bin/env bash
# Controlled S3 ablation: is bidirectional L_path worth it?
#
# Both arms resume the SAME stage-2 checkpoint, so S1 and S2 are shared exactly
# and the only thing that differs is whether L_path also walks f backwards.
#
# Why both arms are re-run rather than comparing against the existing `morph`:
# stage checkpoints written before --resume_stage existed carry no
# discriminators, so a resumed S3 starts with fresh critics.  Comparing a
# resumed arm against the original run would confound path_bidir with
# discriminator state.  Re-running both keeps that difference common.
#
# Also worth knowing: S1's early stop fires at a data-dependent epoch (morph
# left S1 at 83, morph_bi at 120), and S1 does not constrain the cross paths at
# all — so two runs of identical code can enter S2 from very different places.
# Sharing one S1+S2 is what removes that variable.
#
#   bash configs/run_s3_ablation.sh                    # both arms, one GPU
#   FROM=$FLOWCYCLE_EXPS/adni/checkpoints/morph/stage2.pth bash configs/run_s3_ablation.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

FROM="${FROM:-$FLOWCYCLE_EXPS/adni/checkpoints/morph/stage2.pth}"
[ -f "$FROM" ] || { echo "no stage-2 checkpoint at $FROM"; exit 1; }

arm () {                    # arm <tag> <config>
  local tag="$1" cfg="$2"
  python train.py --config "configs/$cfg.yaml" --tag "$tag" \
      --resume_stage 2 --resume_from "$FROM" "${EXTRA[@]}"
}

EXTRA=("$@")
echo "resuming both arms from $FROM on GPU ${CUDA_VISIBLE_DEVICES:-scheduler/default}"
arm morph_s3ctl    morph    &      # path_bidir = 0
arm morph_bi_s3ctl morph_bi &      # path_bidir = 1
wait
echo "both arms done -> $FLOWCYCLE_EXPS/adni/checkpoints/{morph_s3ctl,morph_bi_s3ctl}"
