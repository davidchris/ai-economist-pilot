# Collection profiling on the training server

Date: 2026-09-15.

## Results

Artifacts: [collection-2ek0mfic](results/collection-2ek0mfic/).

| CUDA configuration | Trial 1 steps/s | Trial 2 steps/s | Mean steps/s |
| --- | ---: | ---: | ---: |
| 1 process | 85.9 | 91.0 | 88.5 |
| 2 processes | 166.5 | 155.9 | 161.2 |
| 4 processes | 254.5 | 270.3 | 262.4 |
| 2 processes, no profiling | 167.3 | 164.2 | 165.8 |

Four processes achieved 1.63 times the mean throughput of two processes under the same CPU quota.
Use four processes for the next bounded CUDA experiments. Longer runs still need fresh measurements.

The profiled two-process mean was 2.7% lower than the control mean.
Shared-server variation and synchronization overhead both can contribute to this difference.

Mean timings for the two-process profiled runs:

| Collection component | Seconds | Share of collection |
| --- | ---: | ---: |
| Policy inference and sampling | 10.17 | 46.0% |
| Environment call | 10.26 | 46.4% |
| Tensor conversion and device transfers | 1.14 | 5.2% |
| Bootstrap | 0.07 | 0.3% |

The remaining collection time includes buffer management and instrumentation outside the named spans.

Inference and the environment call each consume about half of collection time.
Device transfers account for a smaller share.
The measurements do not support treating simulation as the only bottleneck.

| Processes | Mean simulator time, summed across workers | Mean echo round-trip per vector step |
| --- | ---: | ---: |
| 1 process | 14.31 s | 0.123 ms |
| 2 processes | 14.81 s | 0.167 ms |
| 4 processes | 16.75 s | 0.360 ms |

All eight runs passed the runtime checks and checkpoint portability checks.
The largest sampled process RSS sum was 4.01 GiB.
The smallest recorded free-space value was 20.88 GiB.

All four automated tests passed on the training server.
The new test compares seeded trajectories with and without profiling and the echo probe.

Test command (inside the same resource-limited scope):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s pilot -p 'test_*.py' -v
```

## Method

The benchmark uses the existing PyTorch environment and the pinned Foundation simulator.
Taxes remain disabled. A shared recurrent worker policy learns through PPO.
Each run collects 4,096 environment steps with seed 42 and a 128-step rollout.
The one-, two-, and four-process runs use 32, 16, and 8 updates, respectively.
Each trial runs those three configurations, then a two-process control without profiling.
The second trial repeats the same order and seed.

Each experiment retains these limits:

- Two CPU cores of aggregate CPU time.
- 8 GiB RAM and no swap.
- A ten-minute timeout with forced termination after ten additional seconds.
- The existing CUDA allocator limit and 15 GiB disk reserve.

Resource snapshots precede each run. Each run saves compact metrics and one checkpoint.
The benchmark directory is unique. Original CPU and CUDA pilot results remain intact.

## Interpretation of timings

Parent timings separate policy inference, tensor conversion and device transfers, and the environment call.
The policy timing includes action sampling and log probabilities.
Worker timings measure Foundation step/reset and the surrounding validation and observation preparation.
These worker spans overlap across processes. Their sums do not represent elapsed collection time.

The transport probe sends an initial observation-sized payload through each worker connection 128 times.
It includes serialization and scheduling but excludes simulation.
The probe gives a transport baseline. It does not isolate transport cost during active training.
The environment round-trip timing includes worker processing and waiting.

CUDA synchronization at each profiling boundary changes execution timing.
The two-process controls help measure this effect.
Different process counts also change PPO batch size, update count, and trajectories.
The results compare throughput, not policy quality.

## Repeat commands

From the project directory:

```sh
bash pilot/benchmark.sh
```

The script prints its unique results directory.
The full commands and resource limits are in `pilot/benchmark.sh`.
Timing field definitions are in `pilot/README.md`.

## Limitations

These short runs use one seed on a shared server.
Both trials use the same order, so time-dependent server load can affect the comparison.
Initialization and the transport probe are outside training-loop throughput.
The throughput figures include PPO updates, recurrent replay checks, and compact metrics.
Sampled process RSS can count shared pages more than once.
These experiments do not establish economic convergence or long-run throughput.
