# Training pilot

This pilot trains worker policies and a separate tax planner in the Salesforce Foundation economy.
It uses native Linux, Python 3.12, and PyTorch 2.9.1 with CUDA 12.6.

## Scope

- The pinned Foundation scenario has four agents, a 25 × 25 map, and 1,000-step episodes.
- Two subprocesses each run a separate economy. The eight workers share one policy.
- Taxes default to disabled; `--tax-regime fixed-tax` trains the fixed US-federal baseline.
- The policy and value networks each use spatial embeddings, convolutions, dense layers, and an LSTM.
- PPO trains on 128-step rollouts, with 16-step recurrent sequences and two optimization epochs.
- Forty updates produce 10,240 environment steps, or 40,960 individual agent transitions.
- Dense logs are disabled. Runs save periodic, retained checkpoints and stream episode/update metrics.

The environment configuration comes from the upstream phase-one example.
The model initialization and training implementation are new.
The short sequence length, optimizer schedule, worker count, and sample budget are pilot choices.
The original labor-cost curriculum remains active and barely advances during this short run.
This pilot does not establish economic convergence, learned tax quality, or reproduction of the published results.

## Files

| File | Purpose |
| --- | --- |
| `environment.py` | Persistent CPU workers and action/observation checks |
| `model.py` | Recurrent actor/critic and generalized advantage estimation |
| `train.py` | Finite training run, metrics, and checkpoint verification |
| `test_pilot.py` | Return-target, action-mask, and recurrent-reset tests |
| `run.sh` | Ten-minute timeout and uv-based training launcher |
| `checkpoint.py` | Versioned atomic snapshots, retention, and committed log recovery |
| `test_resume.py` | Fresh-process restart and checkpoint failure checks |
| `requirements.txt` | Direct dependencies and the official CUDA wheel hash |
| `../upstream/LOCAL_PATCHES.md` | Source revision and NumPy compatibility patch |

The pilot uses a small PyTorch loop to measure the workload without installing Ray or porting its historical wrappers.
Joint planner training is implemented in the same PyTorch trainer; paper-scale configuration
and economic validation remain future work.

## Training-server environment

Project directory: the repository root.

The initial live inspection found 29 GiB free on root and 49 GiB available RAM.
The RTX 3090 had 24 GiB GPU memory, 34 MiB allocated, and 0% utilization.
The NVIDIA driver was `610.43.02`.
These measurements describe the start of this pilot, not ongoing available capacity.

The isolated `.venv` uses the existing Python 3.12.10 interpreter.
Use `uv sync --locked` to install the runtime and development tools from `uv.lock`.
`pyproject.toml` is the dependency source of truth; the requirements files record the historical pilot.
No existing application environment, service, driver, or system package is modified.

## Repeat a run

From the project directory, run the pytest checks inside the bounded scope:

```sh
systemd-run --user --scope --unit=ai-econ-tests \
  -p CPUQuota=200% -p MemoryMax=8G -p MemorySwapMax=0 \
  timeout --signal=TERM --kill-after=10s 600s \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 uv run --locked --no-sync pytest -v
```

For CUDA training, run:

```sh
systemd-run --user --scope --unit=ai-economist-pilot-cuda \
  -p CPUQuota=200% -p MemoryMax=8G -p MemorySwapMax=0 \
  bash pilot/run.sh cuda
```

For the CPU comparison, run:

```sh
systemd-run --user --scope --unit=ai-economist-pilot-cpu \
  -p CPUQuota=200% -p MemoryMax=8G -p MemorySwapMax=0 \
  bash pilot/run.sh cpu
```

Runs create unique `results/train-*` directories. Explicit `--output` paths must not already exist.
Each command limits the entire process tree to two CPU cores and 8 GiB RAM, with no swap.
The CUDA allocator has a 6 GiB limit. This limit does not cap GPU compute utilization.
The training script requires at least 15 GiB free on the output filesystem before and during training.

## Resuming training

The [checkpoint contract and validation report](../RESUME_VALIDATION.md) describe
the complete saved state and exact tested commands. For example, after a run that
finished update 40:

```sh
systemd-run --user --scope --unit=ai-econ-resume \
  -p CPUQuota=200% -p MemoryMax=8G -p MemorySwapMax=0 \
  bash pilot/run.sh cuda --updates 50 \
  --resume results/your-run/checkpoint-000040.pt
```

`--updates` means total updates, so this executes ten more. The default checkpoint
interval is five updates; `--checkpoint-every` changes it. Final updates always
save. `--keep-checkpoints` defaults to two. Resume creates a unique child directory
and inherits omitted training settings. The launcher selects its first argument's
device; direct `train.py --resume ...` inherits the source device when omitted.
Changing device is supported for loading, but does not preserve identical random
trajectories. Checkpoint/model layout, worker count, PPO settings, environment,
runtime versions, and source fingerprint must be compatible.

Only load trusted local checkpoints. Simulator/RNG snapshots contain pickle data.
Preserve each checkpoint's accompanying JSONL logs: resume copies their committed
prefixes and ignores later records. Old pilot checkpoints lack the required state
and cannot resume. The current budget remains capped at 100 updates.

## Validation and interpretation

Every worker checks action masks and finite observations/rewards.
Every PPO update checks finite loss and gradients.
Before each update, a recurrent replay must reproduce the stored action probabilities within a small numerical tolerance.
This check covers sequence ordering, saved hidden states, and episode resets.

The final checkpoint must load on CPU and reproduce the training device's probabilities and values.
The saved parameter difference must show a nonzero optimizer update.

`summary.json` reports measured throughput, process memory, CUDA allocator peaks, and checkpoint size.
The sampled sum of process RSS can double-count shared pages.
Other workloads can affect timing and whole-device resource measurements.

CPU and CUDA runs use the same seed and configuration.
Floating-point differences and stochastic action selection can produce different trajectories.
Their speed comparison is a practical workload measurement, not a bitwise-equivalent benchmark.
The first CUDA iterations include GPU initialization overhead.

## References

- [Salesforce Foundation source](https://github.com/salesforce/ai-economist/tree/a84d5f3fdcabb207d9fde7754d34906903b3e184)
- [Upstream training configuration](https://github.com/salesforce/ai-economist/blob/a84d5f3fdcabb207d9fde7754d34906903b3e184/tutorials/rllib/phase1/config.yaml)
- [Upstream recurrent models](https://github.com/salesforce/ai-economist/blob/a84d5f3fdcabb207d9fde7754d34906903b3e184/tutorials/rllib/tf_models.py)
- [Official PyTorch CUDA installation instructions](https://pytorch.org/get-started/previous-versions/)

## Collection profiling

Run the bounded benchmark on the training server from the project root:

```sh
bash pilot/benchmark.sh
```

The script creates a unique `results/collection-XXXXXXXX` directory and preserves
existing results. It runs two sequential trials of one, two, and four CUDA
environment processes, plus a two-process unprofiled control in each trial.
Each run collects 4,096 environment steps with the same seed and 128-step rollout.
Different process counts change the PPO batch size, update count, and trajectories.
The benchmark compares throughput. It does not compare learning quality.
Each run has its own two-core, 8 GiB, no-swap systemd scope and ten-minute timeout.
Resource snapshots precede each run, and training enforces the 15 GiB disk reserve.

For a single run, add `--profile-collection` to `pilot/train.py` inside the
same resource-limited scope. Timing fields appear in `metrics.jsonl` and
`summary.json`:

- `observation_to_device_seconds`, `action_to_cpu_seconds`, and
  `done_to_device_seconds`: tensor conversion and host/device copies.
- `policy_seconds`: actor/critic inference, action sampling, and log probabilities.
- `environment_seconds`: the full parent-side environment call.
- `bootstrap_seconds`: final observation conversion and value inference.
- `worker_roundtrip_seconds`: action sends through all result receives. This
  includes simulation, serialization, transport, and scheduling waits.
- `observation_stack_seconds`: parent-side observation concatenation, nested
  within `environment_seconds`.
- `simulator_sum_seconds`: summed worker wall time in Foundation step/reset.
- `worker_processing_sum_seconds`: summed worker processing time, including
  simulation, validation, metrics, and packing before sending results.
- `worker_processing_max_seconds`: sum of the slowest worker processing time
  at each vector step. This excludes send/receive and scheduling outside that span.

Worker spans overlap and can include CPU scheduling delays. Do not add them to
parent spans or treat round-trip minus simulator time as a pure IPC measurement.
`transport_probe` separately measures 128 echoes of an initial observation-sized
payload without simulation or observation stacking. It includes serialization
and scheduling. The probe gives a transport baseline, not an exact attribution of live IPC.
The probe does not advance the economy or consume its random stream.

CUDA synchronizes at profiling boundaries to attribute asynchronous work.
This changes execution timing. The unprofiled controls help measure this effect.
The rollout total also includes bookkeeping and instrumentation outside the
named spans. The transport probe is outside training-loop throughput.

## Held-out evaluation

Worker policies support free-market and fixed-tax training; joint checkpoints
support frozen worker–planner held-out evaluation. See [evaluation contract and commands](../EVALUATION_VALIDATION.md).
Run `pilot/evaluate.py --checkpoint PATH --seeds 10000 10001` inside the documented
bounded scope. It evaluates complete episodes on CPU with frozen weights, fresh
recurrent state, full labor costs, and the checkpoint's tax regime.
Evaluation reports productivity, equality, welfare, returns, taxes, and uncertainty
across seeds. Train separate policies in the two regimes and use common held-out
seeds to compare them. Historical checkpoints predate the new source fingerprint
and remain preserved but incompatible; no migration is provided.


## Joint worker–planner training

Use `--phase2-from PATH` with a **current-format free-market checkpoint** to start
phase two. It preserves worker weights and Adam state, creates a separate recurrent
planner and optimizer, enables Foundation's learned taxes and tax annealing, and
uses full labor costs. Both policies train in the same economies. The transition
starts fresh episodes, recurrent state and phase counters. `--updates` is the total
within that phase; use `--resume` for exact update-boundary continuation thereafter.

The planner has seven masked categorical action heads, public per-worker inputs,
256-unit policy/value LSTMs, learning rate 1e-4, and a linear entropy schedule from
2.0 to 0.125 over 50 million aggregate phase-two environment steps. Worker entropy
remains 0.025. Both policies retain numerical, recurrent-replay and CPU portability
checks. Format-v2 checkpoints save all new policy and schedule state.

See [planner contract, commands and measurements](../PLANNER_VALIDATION.md).
Historical checkpoints cannot be migrated. The evaluator now runs both frozen
policies from compatible joint checkpoints; retain the phase-one checkpoint for
held-out seed validation. See the [budget/observation audit and bounded experiment
design](../PAPER_RECONCILIATION.md) before longer runs.
