# Held-out worker and joint planner evaluation

Current checkpoint format is version 2; see [joint planner validation](PLANNER_VALIDATION.md).
The measurements below describe the earlier milestone. Historical checkpoints
remain preserved but fail the current source/version compatibility checks.

Implemented locally, 2026-09-17.

## Joint evaluation update (2026-09-19)

`pilot/evaluate.py` now accepts `learned-tax` snapshots and loads both frozen
policies from the same checkpoint. Each episode starts fresh worker and planner
recurrent states. Both run on every environment step, with Foundation enforcing
seven independent bracket masks and no-ops between tax decisions. Sampled mode
samples each head; greedy mode takes each head's argmax separately. Finite policy
outputs/rewards, masks, worker/planner observation shapes and planner action sizes
are checked. Neither optimizers nor saved economy/recurrent states are used.

Evaluation removes labor and tax annealing for all regimes. For learned taxes,
this exposes the full 0–100% rate range immediately, including actions that a
short checkpoint may never have trained with. It is deployment at full tax
support, not evaluation under the saved curriculum ceiling. No tax or economic
mechanics were changed. The worker/planner architecture and training source
fingerprint are unchanged by this evaluator-only implementation.

Metadata schema 2 records whether the planner ran, full-rate policy, and training
seeds from both phases. Joint evaluation needs the compatible phase-one checkpoint
at its saved `phase2_from` path to verify seed exclusion; it does not need its
training logs. Missing provenance fails closed. This also handles a phase-two
seed/worker-count override. Keep that source checkpoint with joint runs.

Each episode adds `planner_return` (null for baselines) and `tax_schedules`: ten
entries containing the effective seven rates after Foundation processes the first
step of each period, at steps 1, 101, …, 901. The legacy `marginal_rates` field
remains constant for baselines and is null for learned taxes, where one vector
would be misleading. The summary includes planner-return mean/SD/SEM for joint
policies. Source checkpoint hashes and evaluator fingerprints remain recorded.

Compare learned, free-market and fixed-US policies using the same held-out seeds,
action mode and full-cost settings. Episode-seed pairing does not guarantee
identical future random shocks once trajectories diverge. Within-policy SEM is
not uncertainty across independently trained models. The current fixed-tax
baseline starts from scratch: report training budgets and curricula separately.
See [paper reconciliation and bounded design](PAPER_RECONCILIATION.md).

Run inside the required scope (use a new output directory):

```sh
systemd-run --user --scope --unit=ai-econ-joint-eval \
  -p CPUQuota=200% -p MemoryMax=8G -p MemorySwapMax=0 \
  timeout --signal=TERM --kill-after=10s 600s \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/ai-econ-uv-cache \
  uv run --no-sync python pilot/evaluate.py \
  --checkpoint results/YOUR-JOINT-RUN/checkpoint-000100.pt \
  --seeds 10000 10001 --action-mode sample
```

Use the same command with each baseline checkpoint. Keep all training and
evaluation invocations under the resource constraints in `AGENTS.md`.

## Contract and scope

Train separate worker policies with `--tax-regime free-market` (the default) or
`--tax-regime fixed-tax`. The fixed baseline explicitly selects Foundation's
`us-federal-single-filer-2018-scaled` model, uses the upstream scaled US bracket
cutoffs, and disables tax annealing so rates remain 10%, 12%, 22%, 24%, 32%, 35%,
and 37%. Bracket spacing alone does not select this model. Foundation still
collects taxes and redistributes revenue equally; its mechanics are unchanged.
Worker labor warmup remains active during training. No planner policy is added.

`pilot/evaluate.py` loads a compatible trusted format-v2 checkpoint onto CPU.
It uses only its model and configuration, creates a fresh economy and zero
recurrent state for every supplied seed, and runs one complete 1,000-step
episode per seed. It never resumes the checkpoint's unfinished economy or
updates model weights. Training runs and their logs are not modified.
There are 1–20 distinct positive evaluation seeds; seeds assigned to the
checkpoint's training workers are rejected. Use the same held-out seed list for
both separately trained baselines. This partitions initial RNG seeds, not a
separate map/skill dataset: the upstream fixed map and skill configuration remain.

Evaluation uses full labor cost (`energy_warmup_constant=0`) and disables tax
annealing, producing a common stationary objective across policies and seeds.
This differs intentionally from training's evolving labor curriculum. A short
pilot trained under nearly zero labor costs is not expected to perform well.
The evaluation tax regime always comes from the checkpoint; evaluating a
free-market policy under changed taxes is not a trained fixed-tax baseline.

The default action mode samples the masked categorical policy. `--action-mode
greedy` selects the highest-probability valid action and is recorded separately.
Python, NumPy, Torch, and Foundation are seeded for each episode. CPU evaluation
is repeatable in the matching runtime; cross-device/runtime equality is not
promised. CUDA-trained matching checkpoints can be loaded onto CPU.

Each unique output directory contains:

- `config.json`: absolute checkpoint path and SHA-256, training progress and
  worker seeds, evaluator/source fingerprints, effective environment, evaluation
  seeds, device, action mode, and labor policy.
- `episodes.jsonl`: full episode returns, Foundation productivity, equality and
  equality-times-productivity welfare, collected taxes, bracket cutoffs/rates,
  labor weight, and upstream metrics. Undefined upstream diagnostics become
  JSON null; required economic metrics and rewards must be finite.
- `summary.json`: episode/step counts and means, sample standard deviations,
  and standard errors across episode seeds. Single-episode uncertainty is null.
  These measure variation across evaluation seeds for one policy, not uncertainty
  across independently trained policies.

Explicit output directories must be new. A failed/interrupted evaluation can
leave partial episode logs without a summary; rerun in a new directory. No
evaluation resume is implemented. At least 15 GiB free is required before and
between episodes. Use the scope and timeout below for every run.

## Compatibility

Adding the tax configuration changes the existing strict source fingerprint.
Historical checkpoints and results remain intact, but checkpoints predating
this implementation are rejected for both resume and evaluation. No migration
or bypass of the exact-resume compatibility guard was introduced. New fixed-tax
checkpoints include the selected regime; resume inherits it and rejects changes.
The current checkpoint format and full worker/planner/RNG restart-state contract are version 2.

## Commands

Recheck disk, RAM, load and GPU availability before experiments:

```sh
df -h .
free -h
uptime
nvidia-smi
```

Train each baseline in its own new directory (repeat for free-market):

```sh
systemd-run --user --scope --unit=ai-econ-fixed-train \
  -p CPUQuota=200% -p MemoryMax=8G -p MemorySwapMax=0 \
  env UV_CACHE_DIR=/tmp/ai-econ-uv-cache \
  bash pilot/run.sh cpu --tax-regime fixed-tax --workers 1 --updates 5 \
  --output results/fixed-training
```

Evaluate a checkpoint:

```sh
systemd-run --user --scope --unit=ai-econ-evaluate \
  -p CPUQuota=200% -p MemoryMax=8G -p MemorySwapMax=0 \
  timeout --signal=TERM --kill-after=10s 600s \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/ai-econ-uv-cache \
  uv run --no-sync python pilot/evaluate.py \
  --checkpoint results/fixed-training/checkpoint-000005.pt \
  --seeds 10000 10001 --output results/fixed-evaluation
```

Validation commands:

```sh
UV_CACHE_DIR=/tmp/ai-econ-uv-cache uv run --no-sync ruff check pilot
UV_CACHE_DIR=/tmp/ai-econ-uv-cache uv run --no-sync ruff format --check pilot
UV_CACHE_DIR=/tmp/ai-econ-uv-cache uv run --no-sync ty check

systemd-run --user --scope --unit=ai-econ-evaluation-final-tests \
  -p CPUQuota=200% -p MemoryMax=8G -p MemorySwapMax=0 \
  timeout --signal=TERM --kill-after=10s 600s \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/ai-econ-uv-cache \
  EVALUATION_TEST_RESULTS=$PWD/results/evaluation-final-20260917 \
  RUN_CUDA_RESUME_TEST=1 uv run --no-sync pytest -v
```

Use a new artifact directory for another validation run, or omit
`EVALUATION_TEST_RESULTS` to use pytest temporary directories.

## Measured acceptance results

Preflight: 56 GiB disk free, 51 GiB available RAM, and load averages
1.12/0.75/0.47. The host GPU query reported an idle RTX 3090 at 34 MiB.
The initial sandbox-only NVIDIA query could not access the driver; host access
confirmed availability. The test scope was verified at a two-core quota,
8,589,934,592-byte memory limit, and zero permitted swap.

Acceptance artifacts are preserved in `results/evaluation-final-20260917/`
(20 MiB). Each baseline trained for two 16-step updates with one worker economy
and one PPO epoch. Each was evaluated on seeds 10000 and 10001, then repeated in
a fresh process: all episode records matched byte-for-byte. The evaluation tests
also compared model parameters before/after inference and verified no gradients
were produced. Source checkpoint hashes remained unchanged.

| Two-seed mean | Free market | Fixed US-federal |
| --- | ---: | ---: |
| Productivity | 412.867765 | 447.940230 |
| Equality | 0.783530 | 0.749563 |
| Foundation welfare | 85.635721 | 84.932642 |
| Collected taxes | 0 | 83.227293 |
| Mean worker return | -91.535422 | -88.376559 |

These are acceptance-test outputs from 32-step training runs, not evidence of
convergence or a policy ranking. Welfare is Foundation's reported metric: episode equality multiplied by
episode productivity, divided by four workers. Episode logs contain the original metrics and summaries include sample
standard deviation and standard error.

A controlled-income test independently verified that the seven fixed rates
produce nonzero tax collection and equal lump-sum redistribution, conserving
coins; the free-market control collected zero. Fixed-tax uninterrupted training
matched one update plus a fresh-process resume exactly in model, optimizer,
recurrent, RNG, progress, and restored simulator state. Changing regimes on
resume and reusing training seeds for evaluation were rejected. Existing output
directories were preserved.

The initial development test run was stopped after a test compared raw simulator
pickle bytes instead of restored state; serialization byte equality is not the
restart contract. The corrected test uses the existing full-state comparison.
The partial development artifacts remain in `results/evaluation-validation-20260917/`.

Final validation: **all 16 pytest cases passed in 162.90 seconds**, including
the opt-in four-worker CUDA save/resume smoke test and the pre-existing CPU
restart/interrupt/atomic-publication tests. Ruff lint, Ruff format verification,
ty, and `git diff --check` passed. The final test scope is inactive/dead,
no task training/evaluation/worker processes remain, GPU usage returned to
0% / 34 MiB, and disk free is 56 GiB.

## Joint validation results (2026-09-19)

Ruff check, Ruff format check and ty check passed using the existing `.venv`:

```sh
UV_CACHE_DIR=/tmp/ai-econ-uv-cache uv run --no-sync ruff check pilot
UV_CACHE_DIR=/tmp/ai-econ-uv-cache uv run --no-sync ruff format --check pilot
UV_CACHE_DIR=/tmp/ai-econ-uv-cache uv run --no-sync ty check
systemd-run --user --scope --unit=ai-econ-joint-eval-tests \
  -p CPUQuota=200% -p MemoryMax=8G -p MemorySwapMax=0 \
  timeout --signal=TERM --kill-after=10s 600s \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/ai-econ-uv-cache \
  uv run --no-sync pytest -v
```

The full run completed in 258.18 seconds: 19 passed, two opt-in CUDA tests skipped,
and one test-helper failure (plain `dict` compared with PyTorch `OrderedDict`).
The greedy trajectories and tax rates already matched; the weight-comparison
fixture was corrected to preserve the state-dictionary type. Its targeted rerun
passed in 14.72 seconds, using the same scope properties and environment above,
unit `ai-econ-joint-eval-greedy`, and command:
`uv run --no-sync pytest -v pilot/test_evaluate.py::test_joint_greedy_tax_schedule`.
Thus all 20 CPU cases passed across the full run and corrected targeted rerun.
No evaluator implementation change was needed after the full run.

Coverage includes all three regimes' fresh-process sampled evaluation equality,
two complete held-out episodes each, frozen weights and source-file hash,
full labor costs, ten tax schedules per episode, finite metrics and seed rejection.
Joint pretraining uses seed 99 and phase two seed 42 to verify both exclusions.
The greedy check forces distinct bracket rates (5% through 35%), verifies all ten
periods, repeats with fresh state, and rejects a mismatched planner observation
layout. Existing exact CPU worker/joint restart, fixed-tax restart, checkpoint
failure/recovery and profiling tests passed.

The three evaluation JSON reports are retained in ignored
`results/joint-eval-validation-20260919-lr3l7ute/{free-market,fixed-tax,learned-tax}/`.
These are copied reports; their metadata points to the original pytest temporary
checkpoints. They are short engineering acceptance runs, not trained economic
comparisons. Each reports 2,000 evaluation environment steps.

Before validation: 52 GiB disk free, 53 GiB available RAM, load 1.27/0.95/1.34.
`nvidia-smi` could not communicate with the driver; CUDA tests were not enabled.
After validation: 52 GiB free, 53 GiB available RAM, load 2.97/1.66/1.46.
Both scopes are inactive/dead and no trainer/evaluator/pytest/spawned-worker or
resource-tracker processes remain. No driver or other service was changed.
