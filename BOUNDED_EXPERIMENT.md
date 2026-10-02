# Bounded workflow validation — 2026-09-19

Status: **complete**. This executed the bounded design in
[PAPER_RECONCILIATION.md](PAPER_RECONCILIATION.md), not paper reproduction or a
convergence experiment. Existing uncommitted implementation work, historical
results, and pinned upstream source are preserved.

Artifacts are in ignored `results/bounded-20260919/`. Each segment uses a new
directory, keeping its parent logs and checkpoints. `commands.jsonl` records
subsequent commands; `*.resources` records disk, RAM, load and GPU checks before
each invocation; `*.log` and `orchestration.log` retain execution and review results.
The orchestration script is preserved as `results/bounded-20260919/orchestrate.py`
(original invocation: `/tmp/ai-econ-bounded.py`).

## First segment and resource review

Initial sandboxed GPU recheck: `nvidia-smi` could not communicate with the driver.
CPU selected. Subsequent resource checks outside the sandbox (including the
17:20:39 pre-test check) showed an idle RTX 3090, driver 610.43.02, 34 MiB used
out of 24,576 MiB. The initial error was sandbox-specific, not evidence of host
GPU unavailability. CPU was retained for consistent device/restart trajectories;
no driver or service changes. Initial resources: 43 GiB disk free, 40 GiB RAM
available, load averages 8.78/6.08/4.17. System swap was almost full, but each
experiment scope permits zero swap.

```sh
systemd-run --user --scope --unit=ai-econ-bounded-first-20260919 \
  -p CPUQuota=200% -p MemoryMax=8G -p MemorySwapMax=0 \
  env UV_CACHE_DIR=/tmp/ai-econ-uv-cache bash pilot/run.sh cpu \
  --workers 4 --seed 42 --horizon 128 --sequence-length 16 --epochs 2 \
  --updates 20 --checkpoint-every 5 --keep-checkpoints 2 \
  --output results/bounded-20260919/free-market-42-020
```

`pilot/run.sh` supplies the 600-second timeout and single-thread OpenMP/OpenBLAS/MKL
settings. Live scope properties verified: `CPUQuotaPerSecUSec=2s`,
`MemoryMax=8589934592`, `MemorySwapMax=0`. The first sandboxed invocation could
not access the user systemd bus; the same command succeeded with sandbox
escalation, retaining all scope limits.

First segment passed: 20 updates, 10,240 environment steps, 40,960 worker
transitions, eight completed episodes. Training took 59.791 s (171.262 steps/s),
79.530 s total. Sampled peak summed process RSS was 3,050,663,936 bytes
(2.841 GiB; shared pages can be double-counted). Final checkpoint was
29,613,403 bytes (28.242 MiB); two checkpoints retained. CPU reload and forward
agreement passed, worker parameter L2 change was 6.27388, and maximum recurrent
replay error was zero. All action-mask, finite observation/reward/loss/gradient
checks passed. Post-segment disk free remained 43 GiB, available RAM 41 GiB;
load was 10.34/7.63/5.00. The segment was well below the timeout and memory limits.

## Protocol

Training seeds: 42, 142, 242. Each regime/seed ends at 100 updates, reached through
20/40/60/80/100 update targets. Every segment has its own two-core/8-GiB/no-swap
scope and 600-second timeout. Resume inherits saved rollout settings and strict
source compatibility; training source is unchanged throughout this experiment.
Each segment must pass checkpoint reload, replay, finite-value and disk checks
before its child starts. Free-market and fixed-US policies train from scratch;
each joint policy starts from its corresponding final free-market checkpoint.
Joint pretraining is accounted for separately.

Select final update-100 checkpoints and record their SHA-256 values before
held-out evaluation. Evaluate sampled actions on common seeds 10000–10009,
two complete 1,000-step episodes per invocation. Preserve per-seed records,
worker returns and learned tax histories. Summarize paired differences separately
for each training seed; do not conflate episode variation with training variation.

## Validation commands

```sh
UV_CACHE_DIR=/tmp/ai-econ-uv-cache uv run --no-sync ruff check pilot
UV_CACHE_DIR=/tmp/ai-econ-uv-cache uv run --no-sync ruff format --check pilot
UV_CACHE_DIR=/tmp/ai-econ-uv-cache uv run --no-sync ty check
# Full pytest runs with the same scope/timeout/thread limits as experiments:
uv run --no-sync pytest -v
```

Ruff lint, formatting and ty passed before continuation. Full pytest passed: **20 passed, 2 opt-in CUDA tests skipped in 315.18 seconds**.
This includes exact worker/joint CPU restart, phase transfer, atomic checkpoint
failure/recovery, sampled evaluation repeatability for all three regimes, frozen
weights and source hashes, and tax-mechanics checks.

## Interpretation and reproduction prerequisites

This validates the longer workflow only. Joint training receives 51,200 phase-two
steps plus 51,200 worker-pretraining steps per seed; each baseline receives
51,200 steps. The fixed baseline starts from scratch with labor warmup, whereas
joint phase two uses full labor cost. These are not matched treatment comparisons.
Evaluation uses full labor cost and exposes learned taxes to the entire 0–100%
range, beyond short-run training support.

Before any paper reproduction, implement and validate full spatial planner
observations with a suitable map encoder, matched baseline phase transitions,
and explicitly scaled curricula. These require new training and compatibility
contracts; do not retrofit existing checkpoints or silently alter their schedules.
The current experiment deliberately preserves existing mechanics and restart
semantics. Budget and curriculum differences preclude economic convergence or
paper-equivalent policy claims.

## Completed training

All nine phase runs reached 100 updates, 51,200 environment steps, 204,800 worker transitions and 48 completed episodes plus partial episodes. There were 45 separately bounded segments. Joint runs add 51,200 planner transitions each. Total training across the three regimes and three seeds was 460,800 environment steps and 1,843,200 worker transitions.

| Regime | Seed | Sum training seconds | Sum trainer total seconds | Peak sampled RSS (GiB) | Final checkpoint (MiB) | Worker replay max error | Planner replay max error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| free-market | 42 | 217.14 | 278.59 | 2.885 | 31.801 | 0 | — |
| free-market | 142 | 218.95 | 266.69 | 2.888 | 31.807 | 0 | — |
| free-market | 242 | 221.49 | 267.41 | 2.883 | 31.853 | 0 | — |
| learned-tax | 42 | 330.56 | 382.35 | 2.966 | 47.440 | 0 | 1.91e-06 |
| learned-tax | 142 | 366.05 | 425.01 | 2.970 | 47.472 | 0 | 1.91e-06 |
| learned-tax | 242 | 364.24 | 425.60 | 2.919 | 47.438 | 0 | 1.91e-06 |
| fixed-tax | 42 | 225.80 | 286.78 | 2.819 | 31.825 | 0 | — |
| fixed-tax | 142 | 229.02 | 281.57 | 2.888 | 31.866 | 0 | — |
| fixed-tax | 242 | 275.49 | 341.09 | 2.882 | 31.823 | 0 | — |

Training seconds are per-invocation loop times summed across segments. The trainer’s cumulative rollout/update counters and RSS peak are inherited on resume and must not be summed across segments. Total seconds include trainer startup/checkpoint verification, but exclude outer launcher startup/shutdown. RSS is sampled summed process RSS and can double-count shared pages.

Longest trainer invocation: 88.04 s; largest sampled RSS: 2.970 GiB. Every retained directory has two checkpoints. Worker/planner parameter changes were nonzero in every applicable segment. No invalid-action, nonfinite-value, replay or compatibility gate failed.

## Saved curriculum audit

Read-only final snapshot inspection (`curriculum-audit.json`) found all economies
at episode step 800, after 12 complete episodes each. Labor weights (multipliers
of the configured full labor cost) were:

| Regime | Seed 42 range | Seed 142 range | Seed 242 range |
| --- | ---: | ---: | ---: |
| Free market | 0.113877–0.137655 | 0.102642–0.114143 | 0.128117–0.146833 |
| Fixed US tax | 0.089444–0.098955 | 0.132205–0.136447 | 0.110148–0.126197 |
| Joint | 1 | 1 | 1 |

Every joint economy had tax ceiling 0.112. With 0.05 rate discretization, training
still supported only 0%, 5% and 10% rates (plus masked no-op). The saved planner
entropy coefficient for the next update was 1.99808, versus its initial 2.0.
Free-market taxes were disabled; fixed-US rates were unchanged. These observed
curricula are far from matched full-cost treatment comparisons. Held-out evaluation
uses labor weight 1 for all regimes and exposes the planner to 0–100% rates.

The audit used the same systemd/timeout/thread limits as training, with
`uv run --no-sync python /tmp/ai-econ-audit.py`. The first audit invocation lacked
the upstream Python import path and stopped before inspecting simulator state;
the corrected read-only invocation passed. Both logs are preserved. This did not
change training, checkpoints, or evaluation.

## Held-out results

All values below use sampled actions and full labor cost. Within each row, SD describes the ten held-out episode seeds for one trained policy.

| Regime | Training seed | Welfare mean ± SD | Productivity mean ± SD | Equality mean ± SD | Taxes mean ± SD | Mean worker return ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| free-market | 42 | 15.0468 ± 5.0539 | 155.5273 ± 57.4257 | 0.4122 ± 0.1483 | 0.0000 ± 0.0000 | -143.1805 ± 3.6664 |
| free-market | 142 | 164.7712 ± 53.8201 | 828.4496 ± 184.3745 | 0.7824 ± 0.1491 | 0.0000 ± 0.0000 | -76.5075 ± 7.7114 |
| free-market | 242 | 85.6422 ± 46.6750 | 617.9729 ± 201.5489 | 0.5305 ± 0.1892 | 0.0000 ± 0.0000 | -104.4042 ± 11.9363 |
| learned-tax | 42 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | -1.2272 ± 0.2878 |
| learned-tax | 142 | 17.5388 ± 10.7158 | 100.3057 ± 46.2639 | 0.6632 ± 0.1700 | 60.0293 ± 36.7329 | -5.1472 ± 3.7543 |
| learned-tax | 242 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | -1.3427 ± 0.4011 |
| fixed-tax | 42 | 10.4176 ± 5.6356 | 82.5920 ± 34.3792 | 0.5256 ± 0.1653 | 13.6053 ± 4.9536 | -127.4692 ± 5.9943 |
| fixed-tax | 142 | 15.6716 ± 7.8049 | 137.3159 ± 63.4084 | 0.4600 ± 0.0694 | 19.7743 ± 8.3141 | -142.4722 ± 6.7021 |
| fixed-tax | 242 | 174.8158 ± 24.0062 | 930.0946 ± 120.9630 | 0.7545 ± 0.0710 | 128.8180 ± 22.3469 | -95.0605 ± 3.9868 |

Paired differences use the same evaluation seed within each training-seed pair. SD is across ten paired episode differences, not a significance test. Full individual differences are in `paired-differences.csv`.

| Contrast | Training seed | Welfare difference ± SD | Productivity difference ± SD | Equality difference ± SD | Tax difference ± SD | Worker return difference ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| learned-tax minus free-market | 42 | -15.0468 ± 5.0539 | -155.5273 ± 57.4257 | 0.5878 ± 0.1483 | 0.0000 ± 0.0000 | 141.9533 ± 3.8584 |
| learned-tax minus fixed-tax | 42 | -10.4176 ± 5.6356 | -82.5920 ± 34.3792 | 0.4744 ± 0.1653 | -13.6053 ± 4.9536 | 126.2420 ± 5.9840 |
| fixed-tax minus free-market | 42 | -4.6293 ± 8.7530 | -72.9354 ± 61.3306 | 0.1135 ± 0.2214 | 13.6053 ± 4.9536 | 15.7113 ± 6.2255 |
| learned-tax minus free-market | 142 | -147.2325 ± 51.8310 | -728.1439 ± 175.4361 | -0.1193 ± 0.2378 | 60.0293 ± 36.7329 | 71.3603 ± 7.4609 |
| learned-tax minus fixed-tax | 142 | 1.8672 ± 14.1269 | -37.0102 ± 84.6614 | 0.2031 ± 0.1809 | 40.2550 ± 38.4288 | 137.3251 ± 8.3466 |
| fixed-tax minus free-market | 142 | -149.0996 ± 57.9702 | -691.1337 ± 229.1647 | -0.3224 ± 0.1793 | 19.7743 ± 8.3141 | -65.9648 ± 11.9275 |
| learned-tax minus free-market | 242 | -85.6422 ± 46.6750 | -617.9729 ± 201.5489 | 0.4695 ± 0.1892 | 0.0000 ± 0.0000 | 103.0616 ± 12.1526 |
| learned-tax minus fixed-tax | 242 | -174.8158 ± 24.0062 | -930.0946 ± 120.9630 | 0.2455 ± 0.0710 | -128.8180 ± 22.3469 | 93.7178 ± 3.9900 |
| fixed-tax minus free-market | 242 | 89.1736 ± 52.0001 | 312.1218 ± 235.6566 | 0.2240 ± 0.2050 | 128.8180 ± 22.3469 | 9.3438 ± 13.0227 |

Across-training-seed summary: mean ± sample SD of the three policy means (each policy mean averages the same ten episode seeds). Three training seeds give only a small estimate of training variation.

| Regime | Welfare | Productivity | Equality | Taxes | Mean worker return |
| --- | ---: | ---: | ---: | ---: | ---: |
| free-market | 88.4868 ± 74.9027 | 533.9833 ± 344.2336 | 0.5750 ± 0.1891 | 0.0000 ± 0.0000 | -108.0307 ± 33.4841 |
| learned-tax | 5.8463 ± 10.1260 | 33.4352 ± 57.9115 | 0.8877 ± 0.1945 | 20.0098 ± 34.6579 | -2.5724 ± 2.2306 |
| fixed-tax | 66.9683 ± 93.4356 | 383.3342 ± 474.2984 | 0.5801 ± 0.1546 | 54.0659 ± 64.8107 | -121.6673 ± 24.2325 |

## Economic diagnostics and artifacts

Joint seeds 42 and 242 produced zero productivity, welfare and tax collections in
all ten held-out episodes each. Their equality of 1.0 is equality at zero output,
not evidence of successful redistribution. Joint seed 142 produced nonzero
output, but training-seed variation is large across all regimes. Worker returns
are much less negative for joint policies; this must be interpreted together
with the low/zero production and unmatched labor curricula and training budgets.
These outcomes do not establish convergence or a reliable economic ranking.

All joint policies sampled rates spanning 0–100% in evaluation. Of 700 bracket
rates per trained joint policy (ten episodes × ten tax periods × seven brackets),
587, 584 and 584 exceeded the training grid's 10% maximum for seeds 42, 142 and
242 respectively. Their unweighted mean scheduled rates were 48.2929%, 48.2571%
and 48.3857%; these are schedule averages, not effective tax rates or income-
weighted rates. The large expansion of action support is a material limitation.

Preserved files under `results/bounded-20260919/`:

- `selected-checkpoints.json`: nine preselected checkpoint paths and SHA-256 hashes.
- `{free-market,learned-tax,fixed-tax}-{42,142,242}-{020,040,060,080,100}/`:
  unique training segments, retained checkpoints, committed logs and summaries.
- `eval-REGIME-TRAINSEED-EVALSEED/`: paired two-episode evaluation artifacts,
  metadata, complete economic metrics, individual returns and tax schedules.
- `evaluation-records.csv`: 90 primary episode rows, including each worker's return.
- `paired-differences.csv`: all 90 paired contrast rows, three contrasts per
  training/evaluation seed pair. Same initial seeds do not guarantee matched
  future random shocks after policies take different actions.
- `learned-tax-histories.csv`: all 300 learned-policy tax schedules (seven rates
  each), indexed by training seed, evaluation seed and tax-period start.
- `curriculum-audit.json`: final saved economy curricula and phase schedules.
- `commands.jsonl`, resource snapshots and logs: exact bounded invocations and reviews.
- Copies of orchestration/report/audit scripts for reproducibility. Scripts use
  the existing `uv` environment; no trainer or simulator source was modified.

Primary evaluation comprises 90,000 environment steps, distinct from training.
Repeatability episodes are validation only and are excluded from reported means.

## Final verification and resources

All 45 training segments and 45 primary evaluation invocations passed. Three
additional two-episode evaluations of final seed-42 policies (one per regime)
matched their original `episodes.jsonl` files byte-for-byte. These six repeated
episodes are excluded from the 90-episode analysis. All nine selected checkpoint
SHA-256 values remained unchanged after evaluation and repeats. Their compatible
phase-one provenance checkpoints and parent logs remain available.

Ruff lint, Ruff format and ty passed; the full CPU pytest suite passed 20 tests
in 315.18 seconds, with two opt-in CUDA tests skipped. `git diff --check` passed.
No training implementation changes were made for this experiment. Generated
results remain ignored; measured results and commands are documented here.

Final host check at 18:33:54 local: 35 GiB disk free, 41 GiB RAM available,
load 9.91/10.28/10.05. RTX 3090 idle at 0% utilization and 34 MiB allocated.
The experiment occupies about 2.9 GiB. No matching training/evaluation/pytest
processes remained, and no `ai-econ-bnd-*` or `ai-econ-bounded-*` scopes were loaded.
No other service or disk configuration was changed. The first sandboxed GPU
error should not be treated as host GPU unavailability in future checks.
