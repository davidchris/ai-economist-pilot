# Next-agent brief: AI Economist after resumable training

Updated: 2026-09-17. Resumable training is implemented locally on the training server.

## Objective and location

Recreate Salesforce's AI Economist using the existing PyTorch PPO trainer while
preserving Foundation's economic mechanics. The 2020 study is the initial target;
reconcile its configuration with the later upstream defaults before long runs.
Short engineering runs do not establish economic convergence or paper reproduction.

- Project: the repository root, now a local Git repository.
- Python environment: existing `.venv`, managed by `uv`; use `uv sync --locked`.
- Workflow: read `AGENTS.md`. Use Ruff for lint/format, ty for types, pytest for tests.
- `upstream/` is a pinned source archive, not a nested Git repository. See `UPSTREAM.md`.
- Results, caches, `.venv`, and `upstream/` are ignored by the root repository.

## Implemented and validated

`pilot/train.py` now supports `--resume`, periodic atomic checkpoints, bounded
retention, and unique default output directories. `--updates` is the total target.
Explicit output directories must be new. Resume creates a child run and records
its absolute source checkpoint and effective configuration.

Checkpoints capture model/optimizer, progress, parent and worker RNGs, recurrent
states, observations, start masks, and complete Foundation worker object graphs.
Unfinished episodes resume without reset. Component state, episode completions,
partial returns, and labor curriculum progress survive. Future parent-owned
schedule state has an explicit `schedules` field.

Format version 1 rejects legacy checkpoints, source/runtime drift, incompatible
worker counts, PPO rollout settings, observation shapes, and recurrent layouts.
The default is one checkpoint every five updates with two retained; the final
update always saves. Publication uses flush/fsync, atomic rename, and directory
fsync before retention. Previous run directories and historical results are preserved.

Update and episode records stream to JSONL. Checkpoints commit byte offsets;
resume copies only those prefixes into the child. A crash after a log write does
not cause duplicate committed metrics. Keep source checkpoints with their logs.
Summary throughput measures this invocation, while cumulative counters/timing
totals include the committed parent history.

See `RESUME_VALIDATION.md` for the complete contract, measured results, and exact
start/resume/test commands. CPU acceptance compares uninterrupted nine-update
training to four updates plus a fresh-process resume. Restart occurs at episode
step 512, crosses a reset, and ends at step 152 of the next episode. Model,
optimizer, RNG, recurrent, observation, full economy, and progress state agree
exactly. Labor warmup advances from 114 to 179. Timing/RSS are excluded from
equality comparisons. Tests also exercise actual process interruption, torn log
tails, atomic write/rename failures, retention, unique directories, and errors.
The full suite passed all 11 tests in 105.82 seconds, including a four-process
CUDA save/resume smoke test and CPU checkpoint portability. Ruff and ty passed.
Final artifacts are in `results/resume-final-20260917/` (60 MiB).

## Compatibility limits

- These are trusted local pickle snapshots. Do not load untrusted checkpoints.
- Exact continuation is validated on CPU with matching source/runtime. CUDA
  save/resume and CPU checkpoint loading are tested separately; loading on a
  different device does not promise identical stochastic trajectories.
- A source fingerprint deliberately rejects code changes; migration is not implemented.
- The current 100-update cap, 1,000-step episodes, disabled dense logs, and bounded
  checkpoint retention remain. This is not yet an unrestricted long-run trainer.
- Only worker training with taxes disabled exists. Evaluation and planner training
  remain future work. Tax/entropy schedule extensions are not implemented.

## Next work, in order

1. Add held-out evaluation for free-market and fixed-tax worker policies. Verify
   explicit tax-model selection: bracket spacing alone does not select fixed rates.
2. Add a separate planner policy and phase-one-to-phase-two transition, preserving
   worker learning, tax annealing, entropy schedules, and action masks. Extend and
   test the checkpoint contract for every new schedule/policy state.
3. Reconcile paper budgets and observations with upstream defaults, then design
   bounded longer experiments. Do not infer economic findings from pilot success.

Read `README.md`, `pilot/README.md`, `RESUME_VALIDATION.md`, `COLLECTION_PROFILE.md`,
and `PILOT_RESULTS.md` before extending the experiment.

## Operating constraints

The training server has an RTX 3090 and is shared with other services. Before each
experiment check disk, RAM, load, and GPU usage. The latest pre-validation check
showed roughly 58 GiB disk free, 54 GiB available RAM, and an idle GPU using 34 MiB;
these are historical observations, not an ongoing reservation.

Keep each run within two CPU cores, 8 GiB RAM, no swap, and ten minutes, with at
least 15 GiB disk free. Use the existing environment and single-thread BLAS/OpenMP.
Do not alter other services or repurpose other disks. Four environment
processes are appropriate for bounded CUDA experiments if live resources permit.
Confirm experiment processes have stopped before handing off.
At this handoff the test scope is inactive, no training/worker processes remain,
GPU usage is 0% / 34 MiB, and disk free is 56 GiB.
