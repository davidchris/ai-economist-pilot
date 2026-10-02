# Joint worker–planner training

Implemented locally, 2026-09-17. The pinned upstream archive and historical
results are preserved. This is an engineering milestone, not evidence of economic
convergence or reproduction of a paper result.

## Transition and policies

`--phase2-from PATH` starts a new phase-two run from a current-format free-market
checkpoint. It copies the worker model and Adam optimizer, including moments and
step counters. Workers continue learning at 3e-4. The new planner has independent
policy/value towers, each with two 256-unit dense layers, layer normalization and
a 256-unit LSTM, and its own Adam optimizer at 1e-4. Worker towers remain unchanged.

The transition intentionally starts fresh economies, recurrent states, random
streams (using the effective seed), phase counters and logs. The source path and
phase-one progress are recorded. It is not an exact continuation of an unfinished
phase-one economy. It uses full labor costs, as in upstream phase two; the
phase-one labor curriculum is not replayed. Existing worker-only training and
exact resume retain their labor curriculum. Omitted worker count, seed and PPO
sizes are inherited from the phase-one checkpoint.

The planner receives Foundation's flattened nonspatial planner observation plus
all public per-worker observations, concatenated in sorted key order. Foundation
exposes seven action subspaces of size 22 (no-op plus rates 0 through 1 in 0.05
increments). Independent masked categorical distributions produce one action per
bracket. PPO uses their summed log probabilities and entropy, and Foundation's
planner welfare reward. Both policies collect from the same economies and update
after each rollout; worker and planner GAE/minibatches remain separate. Planner
values train on every step, including the no-op steps between tax decisions.

Foundation retains responsibility for tax timing, bracket accounting, collection,
redistribution, and tax annealing `[-100, 0.001]`. Its own masks allow rate changes
only on the first day of each 100-step tax period and initially cap rates at 10%.
Annealing follows each economy's completed episodes, saved in its object graph.
The worker entropy coefficient stays at 0.025. Planner entropy decreases linearly
from 2.0 to 0.125 over 50 million aggregate phase-two environment steps, then stays
at 0.125. The coefficient is selected at the start of each PPO update; snapshots
record the coefficient and step count for the next update. This step convention
is explicit; these runs are not an RLlib trajectory equivalence claim.

## Checkpoints and compatibility

Format version 2 includes both models, optimizers, recurrent states, observations,
start masks, RNG streams, full Foundation graphs and partial episode returns, phase counters, entropy schedule
state and committed log offsets. `--resume` continues either phase at an exact
update boundary without resetting environments. Joint CPU equality is checked
across tax decisions and an episode reset. CUDA validation covers save/resume and
CPU loading/forward agreement, not bitwise CPU/CUDA trajectories.

The existing strict source/runtime fingerprint applies to both resume and phase
transfer. Version 1 and historical-source snapshots remain preserved but cannot
be migrated or used as phase-two sources by this implementation. Generate a new
phase-one checkpoint with this source. Fixed-tax and joint checkpoints are not
valid phase-one sources. `--resume` and `--phase2-from` are mutually exclusive.
Joint held-out evaluation is now implemented; see [evaluation validation](EVALUATION_VALIDATION.md)
for its frozen-policy, full-rate-support and seed-provenance contract.

## Commands

Check live disk, RAM, load and GPU availability first, retaining at least 15 GiB
free. Use new output directories. From the project root:

```sh
systemd-run --user --scope --unit=ai-econ-phase1 \
  -p CPUQuota=200% -p MemoryMax=8G -p MemorySwapMax=0 \
  bash pilot/run.sh cuda --workers 4 --updates 5 --output results/my-phase1

systemd-run --user --scope --unit=ai-econ-phase2 \
  -p CPUQuota=200% -p MemoryMax=8G -p MemorySwapMax=0 \
  bash pilot/run.sh cuda --phase2-from results/my-phase1/checkpoint-000005.pt \
  --updates 5 --output results/my-phase2

systemd-run --user --scope --unit=ai-econ-phase2-resume \
  -p CPUQuota=200% -p MemoryMax=8G -p MemorySwapMax=0 \
  bash pilot/run.sh cuda --resume results/my-phase2/checkpoint-000005.pt --updates 10
```

`--updates` is the total target within the current phase. The existing 100-update,
1,000-step episode and checkpoint retention limits remain.

Validation commands (the cache override keeps tool writes inside the sandbox):

```sh
export UV_CACHE_DIR=/tmp/ai-econ-uv-cache
uv run --no-sync ruff check pilot
uv run --no-sync ruff format --check pilot
uv run --no-sync ty check
systemd-run --user --scope --unit=ai-econ-planner-final-tests \
  -p CPUQuota=200% -p MemoryMax=8G -p MemorySwapMax=0 \
  timeout --signal=TERM --kill-after=10s 600s \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/ai-econ-uv-cache \
  RUN_CUDA_RESUME_TEST=1 \
  PLANNER_TEST_RESULTS=$PWD/results/planner-final-20260917/cpu \
  PLANNER_CUDA_TEST_RESULTS=$PWD/results/planner-final-20260917/cuda \
  uv run --no-sync pytest -v
```

The existing collection transport probe still echoes a worker-observation payload;
it is not a measurement of the full joint payload. Joint profiling additionally
reports `planner_policy_seconds`, including planner tensor/action transfers.

## Measured acceptance results

All **20 tests passed in 290.75 seconds**, including both CUDA smoke tests. Ruff
lint, Ruff format verification, ty and `git diff --check` passed.

Artifacts: `results/planner-final-20260917/` (about 194 MiB). Pre-suite resources
were 59 GiB disk free, 53 GiB available RAM, load averages 6.65/4.54/3.15 and an
idle RTX 3090 using 34 MiB. The live scope reported `CPUQuotaPerSecUSec=2s`,
`MemoryMax=8589934592` and `MemorySwapMax=0`.

- CPU joint training: nine uninterrupted 128-step updates matched four updates
  plus a fresh-process resume to nine bit for bit. The checkpoint was at episode
  step 512; completion crossed a reset and ended at step 152. Both models,
  optimizers, recurrent states, observations, RNGs, schedules, progress, complete
  economies and partial returns matched. Committed metrics matched with timing
  and RSS excluded; an extra uncommitted log record was discarded on resume.
- Worker weights and optimizer state matched the phase-one source exactly at
  transfer. Both policies then learned: the uninterrupted run's parameter L2
  changes were 1.24907 (worker) and 0.46448 (planner). Its training loop took
  13.29 seconds for 1,152 environment steps. This short shared-server measurement
  is not a long-run throughput estimate.
- CUDA joint training: four economies, 32-step rollouts, two updates followed by
  resume to four; final progress was 512 environment steps. Start/resume training
  loops took 2.25/2.10 seconds; total invocations took 13.80/11.79 seconds.
  Both planner and worker checkpoints passed CPU portability/forward agreement.
  Maximum planner recurrent replay error across CPU/CUDA runs was `9.54e-7`.
- Planner input contains 111 nonspatial values and 154 action-mask entries;
  its separate model has 1,282,459 parameters. The resumed CUDA checkpoint was
  22,888,547 bytes. Peak CUDA tensor allocation was 116,465,152 bytes; maximum
  sampled summed training-process RSS was 4,286,332,928 bytes (3.99 GiB).
- Foundation checks verified seven bracket masks, no-op-only intermediate tax
  steps, the 10% initial rate cap, later annealing expansion, and 10 coins of tax
  collected from a known 100-coin income and redistributed equally. Model tests
  cover joint log probabilities, entropy, fully masked action rejection and
  recurrent reset semantics. Invalid phase sources/options are rejected before
  creating output directories. Entropy midpoint and final plateau are checked.

Post-validation: the test scope was inactive/dead; no trainer, pytest, spawned
worker or multiprocessing resource tracker remained. Disk free was 59 GiB,
available RAM 53 GiB, and GPU usage returned to 0% / 34 MiB.
