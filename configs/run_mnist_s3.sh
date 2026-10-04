#!/usr/bin/env bash
# MNIST-PET/CT S3 ablation — three ways to supervise the path, one shared S1+S2.
#
#   abs     D_mix with an absolute target (fake -> 1).  The ADNI baseline.
#   ra      relativistic D_mix.  An intermediate frame can never enter either
#           real set, so `fake -> 1` is unreachable; ranking near the reals is.
#   smooth  no adversarial term.  Weakest arm on both ADNI and horse2zebra;
#           run as the control that keeps the ablation honest.
#
# All arms resume the SAME stage-2 checkpoint, so S3 is the only difference.
#
# Choosing FROM.  On horse2zebra every round-2 arm forked from the latcyc S2,
# which was later shown to have been cut short by an early-stopping criterion
# that was blind to f (see val_monitor in train.py).  That criterion is fixed,
# but the choice of fork point is now an empirical one rather than a default:
# run BOTH mnist_base and mnist_latcyc first, then point FROM at whichever S2
# scored better, and record which was used.
#
#   FROM=$FLOWCYCLE_EXPS/mnist/checkpoints/mnist_latcyc/stage2.pth bash configs/run_mnist_s3.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
FROM="${FROM:-$FLOWCYCLE_EXPS/mnist/checkpoints/mnist_latcyc/stage2.pth}"
GPUS="${GPUS:-0 1}"

if [ ! -f "$FROM" ]; then
  echo "no shared S2 at $FROM -- run round 1 first:" >&2
  echo "  python train.py --config configs/mnist_latcyc.yaml" >&2
  exit 1
fi
echo "forking all arms from $FROM"

arm () {  # arm <gpu> <tag> <extra...>
  local gpu="$1" tag="$2"; shift 2
  echo "[$(date +%H:%M:%S)] start $tag on GPU$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python -u train.py --config configs/mnist_morph.yaml \
      --tag "$tag" --resume_stage 2 --resume_from "$FROM" "$@"
  echo "[$(date +%H:%M:%S)] done  $tag"
}

set -- $GPUS
A="$1"; B="${2:-$1}"

( arm "$A" mnist_morph_ra --path_gan_mode ra
  arm "$A" mnist_morph_smooth --w_path_gan 0 --w_path_smooth 1.0 ) &
P1=$!
( arm "$B" mnist_morph_abs ) &
P2=$!
wait $P1 $P2
echo "done -> $FLOWCYCLE_EXPS/mnist/checkpoints/{mnist_morph_abs,mnist_morph_ra,mnist_morph_smooth}"
