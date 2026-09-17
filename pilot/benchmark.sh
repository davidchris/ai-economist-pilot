#!/usr/bin/env bash
# Each experiment retains the pilot's two-core, 8 GiB, no-swap, ten-minute limits.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
mkdir -p results
output=$(mktemp -d results/collection-XXXXXXXX)
echo "$output"
for repeat in 1 2; do
  for workers in 1 2 4; do
    # Match environment-step budgets at 4,096 steps per run.
    device=cuda
    run="$output/$device-w$workers-r$repeat"
    { date -u; df -B1 .; free -m; uptime; nvidia-smi; } > "$run-resources.txt"
    systemd-run --user --scope --unit="ai-econ-$(basename "$output")-$device-w$workers-r$repeat" \
      -p CPUQuota=200% -p MemoryMax=8G -p MemorySwapMax=0 \
      timeout --signal=TERM --kill-after=10s 600s nice -n 10 \
      uv run --locked --no-sync python pilot/train.py --device "$device" --workers "$workers" \
      --updates "$((32 / workers))" --profile-collection --output "$run" > "$run.log" 2>&1
  done
  run="$output/cuda-w2-control-r$repeat"
  { date -u; df -B1 .; free -m; uptime; nvidia-smi; } > "$run-resources.txt"
  systemd-run --user --scope --unit="ai-econ-$(basename "$output")-control-r$repeat" \
    -p CPUQuota=200% -p MemoryMax=8G -p MemorySwapMax=0 \
    timeout --signal=TERM --kill-after=10s 600s nice -n 10 \
    uv run --locked --no-sync python pilot/train.py --device cuda --workers 2 --updates 16 \
    --output "$run" > "$run.log" 2>&1
  echo "Completed trial $repeat: $output"
done
