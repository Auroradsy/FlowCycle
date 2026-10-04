#!/usr/bin/env bash
# Train the plain CycleGAN host that train.py splits into E/D.
# Only needed if you do not already have $FLOWCYCLE_EXPS/adni/checkpoints/host/last.pth.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
export HOST_TAG="${HOST_TAG:-host}"
python train_host.py "$@"
