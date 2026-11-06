# Training-server pilot results

Date: 2026-09-15. **Both the CUDA and CPU runs passed.**

The training server can run the Foundation simulator and a modern PyTorch learner within a small resource budget.
The CUDA run had 18.4% higher overall throughput than the CPU run.
Experience collection dominated elapsed time, so more GPU compute alone will not solve the throughput limit.

## Experiment

Each run used two independent Foundation economies with four workers each.
The workers shared one recurrent policy. Taxes were disabled, and no planner policy trained.
Each run completed 40 PPO updates, 10,240 environment steps, and 40,960 individual agent transitions.
Ten complete 1,000-step episodes finished per run. Each economy also contributed a final partial episode.

The economic configuration came from the pinned upstream phase-one example.
The pilot used a new PyTorch implementation with separate recurrent policy and value networks.
Its smaller training budget and sequence length differ from the published experiments.
The [pilot instructions](pilot/README.md) describe these choices and the repeat commands.

## Measured performance

| Measurement | RTX 3090 / CUDA | CPU |
| --- | ---: | ---: |
| Training loop | 54.11 s | 64.08 s |
| Total script time, including initialization and verification | 59.03 s | 70.03 s |
| Environment steps per second | 189.25 | 159.80 |
| Experience collection | 46.48 s | 48.54 s |
| PPO update phase | 6.97 s | 15.16 s |
| Maximum sampled sum of process RSS | 2.94 GiB | 1.74 GiB |
| Checkpoint size | 4.47 MiB | 4.46 MiB |

CUDA made the update phase about 2.18 times faster.
Experience collection consumed 85.9% of the CUDA training loop.
That phase includes policy inference, simulator steps, interprocess communication, and data transfer.
This measurement does not isolate the simulator as the sole bottleneck.

One-second GPU samples averaged 5% utilization during the CUDA run, with a maximum of 21%.
The maximum sampled whole-device allocation was 502 MiB, including driver overhead and the pre-existing allocation.
PyTorch reported 90.7 MiB of peak tensor allocation and 134 MiB of peak reserved memory.

The process RSS sum can double-count shared pages.
Late cgroup samples recorded peaks of 2.04 GiB for CUDA and 1.23 GiB for CPU.
Those cgroup snapshots precede process exit and are not final lifetime measurements.

These were sequential runs on a shared server, with one seed and no repeated timing trials.
CPU and CUDA random sampling produced different trajectories.
The ratio is a measured pilot result, not a guaranteed speedup for longer experiments.

## Correctness checks

- All three numerical tests passed: terminal returns, rollout bootstrapping, and action masking with recurrent episode resets.
- Every environment action complied with its current action mask.
- Observations, rewards, losses, and gradient norms remained finite.
- Before each update, recurrent replay reproduced stored action log probabilities.
- The largest replay error was less than `1e-6` in both runs.
- Both optimizers changed the model weights.
- Both checkpoints loaded on CPU. The CUDA checkpoint matched CPU forward probabilities and values within the specified tolerance.
- The installed environment passed `uv pip check` for all 33 packages.

The economies performed actual transactions and construction.
The CUDA run recorded 2,070 trades and 165 house builds across its ten complete episodes.
The CPU run recorded 2,405 trades and 300 house builds.
These totals verify activity. They do not demonstrate policy quality or a difference in learning effectiveness between devices.

## Resource limits and final state

Both runs used systemd scopes with these limits:

- Two CPU cores of aggregate CPU time (`CPUQuota=200%`).
- 8 GiB RAM (`MemoryMax=8G`) and no swap (`MemorySwapMax=0`).
- Ten minutes per run, followed by termination if necessary.
- A 6 GiB PyTorch CUDA allocator limit.
- A 15 GiB minimum free-space reserve, checked before and during training.

The isolated environment remains at `.venv` and occupies approximately 6.0 GiB.
The remote result directory occupies approximately 9.2 MiB.
Final root free space was approximately 22.3 GiB (`df -h` rounds this to 23G).
Temporary download storage returned to its initial level. No persistent package-download cache was created.

Both systemd scopes are inactive. No pilot training or telemetry processes remain.
The GPU returned to 0% utilization and 34 MiB allocated.
Existing services, application environments, and NVIDIA drivers were unchanged.

## Collection profiling follow-up

The [profiling comparison](COLLECTION_PROFILE.md) now separates collection costs and compares one, two, and four CUDA environment processes.
Four processes averaged 262.4 steps/s under the same two-core limit across two short trials.
Resumable training is the next implementation milestone.

## Original recommendation

Keep the training server as the primary training target, with CUDA available.
First profile experience collection and benchmark more environment workers within an agreed CPU budget.
Then validate the full recurrent PPO configuration and add the planner curriculum.

The 3090 has considerable memory and compute headroom in this pilot.
System RAM and rollout throughput need measurement as worker counts increase.
The current storage is sufficient for small checkpoints and compact metrics.
Full episode logs and large published datasets remain outside this pilot's storage budget.

For scale only: 400 million steps at the measured CUDA throughput take about 24.5 days.
That arithmetic uses the two-core pilot configuration and excludes planner training, baselines, evaluation, and multiple seeds.
It is not an estimate for an optimized full-server run.

**This is a successful engineering pilot, not a reproduction of the paper's economic findings.**

## Saved artifacts

Result files are not tracked:

- `results/cuda/`: `summary.json`, `config.json`, `metrics.jsonl`, and `checkpoint.pt`.
- `results/cpu/`: the same four files.
- GPU telemetry: `results/cuda-gpu.csv` and `results/cpu-gpu.csv`.
- Scope samples: `results/cuda-scope.txt` and `results/cpu-scope.txt`.
- [Dependency lock](pilot/requirements.lock.txt) and [upstream revision/patch](UPSTREAM.md).
