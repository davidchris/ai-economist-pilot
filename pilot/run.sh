#!/usr/bin/env bash
# Invoke inside the documented systemd scope so all children share its limits.
set -euo pipefail
cd "$(dirname "$0")/.."
device="${1:-cuda}"
if (($#)); then shift; fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
timeout --signal=TERM --kill-after=10s 600s nice -n 10 uv run --locked --no-sync python pilot/train.py --device "$device" "$@"
