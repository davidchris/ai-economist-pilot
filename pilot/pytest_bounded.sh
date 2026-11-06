#!/usr/bin/env bash
# Run the test suite inside the resource limits required by AGENTS.md.
# Two scopes keep each part well inside the 600-second limit, including on a loaded host.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [[ ! -d upstream/ai_economist ]]; then
  echo "upstream/ is missing. See UPSTREAM.md." >&2
  exit 1
fi

free_gib=$(df --output=avail -BG . | tail -n 1 | tr -dc '0-9')
if ((free_gib < 15)); then
  echo "Only ${free_gib} GiB free; the project requires a 15 GiB reserve." >&2
  exit 1
fi

run_scope() {
  local label=$1
  shift
  systemd-run --user --scope --collect --quiet --unit="ai-econ-pytest-$label-$$" \
    -p CPUQuota=200% -p MemoryMax=8G -p MemorySwapMax=0 \
    timeout --signal=TERM --kill-after=10s 600s \
    env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    uv run --no-sync pytest -q "$@"
}

run_scope paper pilot/test_paper.py
run_scope rest --ignore=pilot/test_paper.py
