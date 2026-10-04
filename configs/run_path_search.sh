#!/usr/bin/env bash
# Three ways out of the D_mix problem, all resumed from one shared stage-2 so
# that S3 is the only thing that differs.  The baseline they answer to is
# morph_s3ctl (forward-only, absolute critic): D_B 0.183 / D_A 0.120.
#
#   smooth  drop the adversarial term entirely.  Asks the basic question:
#           is D_mix needed at all, or does smoothness + L_latcyc suffice?
#   ra      relativistic critic.  An in-between frame can never enter either
#           real set, so `fake -> 0` is an unreachable target; ranking near the
#           reals IS reachable.
#   bisep   bidirectional, but each leg gets its own critic, removing the
#           two-opposite-targets conflict that collapsed morph_bi_s3ctl.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
FROM="${FROM:-$FLOWCYCLE_EXPS/adni/checkpoints/morph/stage2.pth}"

arm () {  # arm <gpu> <tag> <config> <extra...>
  local gpu="$1" tag="$2" cfg="$3"; shift 3
  CUDA_VISIBLE_DEVICES="$gpu" python train.py --config "configs/$cfg.yaml" --tag "$tag" \
      --resume_stage 2 --resume_from "$FROM" "$@"
}

arm 0 morph_smooth   morph    --w_path_gan 0 --w_path_smooth 1.0 &
arm 0 morph_ra       morph    --path_gan_mode ra &
arm 1 morph_bi_sep   morph_bi --path_critics separate &
wait
echo "done -> $FLOWCYCLE_EXPS/adni/checkpoints/{morph_smooth,morph_ra,morph_bi_sep}"
