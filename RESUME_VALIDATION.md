# Resumable training validation

Current checkpoint format is version 2; see [joint planner validation](PLANNER_VALIDATION.md).
The measurements below describe the earlier milestone. Historical checkpoints
remain preserved but fail the current source/version compatibility checks.

Implementation and measurements are local to the training server, 2026-09-17.

## Contract

Format version 1 checkpoints are taken only after completed PPO updates. `--updates`
is the total target, including saved updates. Resume creates a new child run; its
`config.json` records the source checkpoint and effective configuration. Training
settings omitted on resume are inherited. Explicit incompatible changes are rejected.

Each checkpoint contains model/optimizer state, counters, four actor/critic hidden/cell
tensors, observations, episode-start masks, Python/NumPy/Torch/CUDA RNG state, and
each worker's complete serialized Foundation object graph, observations, partial
returns, and Python/NumPy RNG state. The graph includes world/agent/component state,
auction orders, replay buffers, episode completions, the labor warmup integrator,
and tax component counters. No independent random generators were found in the
selected Foundation scenario/components. A `schedules` field reserves parent-owned
state for future tax/entropy schedules, which remain unimplemented.

Foundation's episode replay API stores reset/action NumPy states, but does not
restore a full multi-episode curriculum and Python RNG contract. Full object
serialization avoids resetting or replaying an unfinished economy. This is a
trusted-local pickle format: only load checkpoints you trust. The outer Torch
container loads with `weights_only=True`, but simulator and RNG byte blobs are
deserialized with pickle. Python, NumPy, Torch, and source fingerprints must match.
Legacy pilot checkpoints are rejected for resume. CPU loading is supported;
cross-device stochastic equality is not promised.

The default interval is five updates and retention is two checkpoints. Temporary
files are flushed and fsynced before atomic rename and directory fsync. Old
checkpoints are removed only after publication, within the new run directory.
A hard kill may leave an ignored temporary file in the abandoned run; previous
complete checkpoints remain usable. Retention bounds published checkpoints in
the active child, not the user's archive of parent runs.

Metrics and episode records stream to JSONL, with no growing in-memory history.
Checkpoints store durable byte offsets for both logs. Resume copies only those
prefixes to the child, discarding speculative or torn records after the selected
checkpoint without changing the parent. Keep the checkpoint together with its
run's logs. The current 100-update experiment cap and fixed 1,000-step episodes
bound disk logs and Foundation replay history; this is not yet a long-run trainer.

## Commands

Install and check code:

```sh
uv sync --locked
uv run --no-sync ruff check pilot
uv run --no-sync ruff format --check pilot
uv run --no-sync ty check
```

Start and resume bounded CUDA runs (use a new explicit directory for each):

```sh
systemd-run --user --scope --unit=ai-econ-start \
  -p CPUQuota=200% -p MemoryMax=8G -p MemorySwapMax=0 \
  bash pilot/run.sh cuda --workers 4 --updates 5 \
  --checkpoint-every 1 --output results/my-start

systemd-run --user --scope --unit=ai-econ-resume \
  -p CPUQuota=200% -p MemoryMax=8G -p MemorySwapMax=0 \
  bash pilot/run.sh cuda --updates 10 \
  --resume results/my-start/checkpoint-000005.pt
```

Full pytest acceptance suite, retaining CPU and CUDA comparison artifacts:

```sh
systemd-run --user --scope --unit=ai-econ-resume-final-tests \
  -p CPUQuota=200% -p MemoryMax=8G -p MemorySwapMax=0 \
  timeout --signal=TERM --kill-after=10s 600s \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  RESUME_TEST_RESULTS=$PWD/results/resume-final-20260917 \
  CUDA_TEST_RESULTS=$PWD/results/resume-final-20260917 \
  RUN_CUDA_RESUME_TEST=1 \
  uv run --locked --no-sync pytest -v
```

The artifact directory must be new. Omit both result environment variables for
pytest-managed temporary artifacts. CUDA is skipped unless `RUN_CUDA_RESUME_TEST=1`.

## Measured results

All **11 pytest cases passed in 105.82 seconds** in the scope above. Ruff lint,
Ruff format verification, ty, and shell syntax checks also passed.

- CPU: nine uninterrupted updates matched four updates plus a fresh-process
  resume to update nine exactly. Each run used one economy, 128-step rollouts,
  and one PPO epoch. Final progress was 1,152 environment steps and 4,608 agent
  transitions. Restart was at episode step 512; completion crossed one reset and
  ended at step 152. The labor warmup integrator advanced from 114 to 179.
  The full comparison, including subprocess startup and rejection checks, took
  53.72 seconds. Model, optimizer, RNGs, recurrent state, observations, counters,
  complete worker economy state, and numerical update metrics agreed exactly.
- CUDA: four economies, 32-step rollouts, one epoch, two updates followed by a
  fresh-process resume to four. Final counters were 512 environment steps and
  2,048 agent transitions. Start/resume training intervals took 1.845/1.553 seconds;
  total invocation times including startup were 12.055/10.572 seconds. Both runs
  passed action masks, finite gradients, recurrent replay, CPU loading, and
  CPU/CUDA forward agreement. This smoke test does not assert CUDA trajectory equality.
- Maximum sampled training process RSS was 4,284,432,384 bytes (3.99 GiB).
  Peak CUDA tensor allocation was 91,006,976 bytes; reserved memory was
  123,731,968 bytes. The resumed CUDA checkpoint was 7,361,893 bytes. CPU final
  checkpoints were approximately 9.3 MB. All retained acceptance artifacts total 60 MiB.
- Atomic partial-write and rename failure injection preserved the last checkpoint.
  Retention stayed at two published checkpoints per run. Unique directories,
  legacy/runtime incompatibility, changed worker/sequence/horizon errors, missing
  logs, and simulated torn tails passed. An actual SIGTERM of a training process
  group after update six resumed from update five and produced updates 1–7 once each.

Raw reports and snapshots are in `results/resume-final-20260917/`, including
`comparison.json`, each run's `summary.json`, JSONL metrics, and retained checkpoints.
The earlier eight-test development run is preserved in
`results/resume-validation-20260917/`; its checkpoints predate final source changes
and intentionally fail the final source compatibility check.

Post-run checks: `ai-econ-resume-final-tests.scope` was inactive/dead; no trainer,
pytest, spawned worker, or multiprocessing resource-tracker processes remained.
The GPU returned to 0% utilization and 34 MiB usage. Disk free was 56 GiB.
