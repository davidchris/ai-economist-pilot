# Next-agent brief: AI Economist after productive worker pretraining

Public release preparation (2026-11-06): history was rewritten to remove a personal email
and machine details, so commit hashes changed. Pre-push hooks run Ruff, ty, betterleaks and
`pilot/pytest_bounded.sh`. Measured on this source: 43 passed, 3 opt-in tests skipped, in
206.61 s and 216.58 s across two scopes. Training source is unchanged.

Completed milestone (2026-11-05): see [PRETRAINING_MILESTONE.md](PRETRAINING_MILESTONE.md).
Workers now sustain productive behavior at full labor cost: three training seeds pass
a prespecified gate on fresh episode seeds and through 240k full-cost phase-two steps.
Use `--minibatch-sequences 4 --labor-warmup-constant 40000` for 4.4M steps; the three
checkpoints are listed there. The collapse came from the labor curriculum, not batch
size. Worker observations now include `time`, so earlier checkpoints are incompatible.
No convergence or paper reproduction is established.

Latest result (2026-11-06): see [TAX_COMPARISON.md](TAX_COMPARISON.md). The learned-tax arm
collapsed on all three seeds with a planner still close to random, so this is not evidence
about learned taxes as such. Fixed-tax and free-market stayed productive. Next: find out why
the planner does not learn, starting with a uniform-random planner control. Editing
`pilot/train.py` makes the source fingerprint reject the pretraining checkpoints. NVIDIA
packages were upgraded on 2026-11-05 without a reboot, so `nvidia-smi` fails until one.
Generated results are ignored and all task processes have stopped.

Previous milestone (2026-11-04): see [PAPER_MILESTONE.md](PAPER_MILESTONE.md).
The paper2020 profile implements spatial planner observations, matched transitions
for all three current tax regimes, explicit curriculum settings and bounded
continuation beyond 100 total updates. Checkpoint format is now 3; historical
snapshots are preserved and cannot be migrated. All 28 tests passed, including
three CUDA cases. The matched diagnostic completed 800k training steps and 80
held-out episodes within the resource limits. All three final policies produced
zero output; this collapse already occurred during shared worker pretraining.
Next work is productive worker pretraining with a justified curriculum/batch
budget before another tax-policy comparison. Source and results are archived in
ignored `results/paper-medium-20261104/`; all task processes/scopes stopped.
The entries below describe the previous milestone and its historical measurements.

Updated: 2026-09-19. The three-seed bounded training and held-out evaluation workflow is complete; see BOUNDED_EXPERIMENT.md.

## Commit validation (2026-10-03)

The completed planner, evaluation and bounded-experiment work was reviewed for
commit without changing training or simulator source. The following checks passed:

```sh
uv run --no-sync ruff check pilot
uv run --no-sync ruff format --check pilot
uv run --no-sync ty check
git diff --check
```

Ruff reported all 12 Python files already formatted. Pytest was not rerun:
`df -h .` showed only 7.0 GiB free, below the required 15 GiB reserve. No training
or environment tests were launched. The latest full test result remains the
September 19 bounded-experiment validation: 20 passed, two opt-in CUDA tests
skipped in 315.18 seconds. The current resource check found 53 GiB available RAM,
load averages 1.38/1.09/1.03, and an idle RTX 3090 using 34 MiB. These readings
supersede historical resource snapshots for this review only; recheck before runs.
Generated results, the virtual environment and pinned upstream source remain
excluded from Git, and historical artifacts were preserved.

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

## Joint planner milestone

The separate planner policy and phase-one-to-phase-two transition are implemented.
See `PLANNER_VALIDATION.md` for commands, validation, and the exact phase contract.
`--phase2-from` transfers a current-format free-market worker model **and optimizer**
into fresh full-labor-cost phase-two economies. Workers keep learning; the separate
256-unit recurrent planner learns seven independently masked tax actions. Foundation
owns tax timing and annealing. Worker entropy stays at 0.025; planner entropy falls
from 2.0 to 0.125 over 50 million aggregate phase-two environment steps.

Format version 2 snapshots include the planner model/optimizer, recurrent state,
observations and start masks, entropy/phase state, both policies' RNG streams and
complete Foundation graphs. `--resume` continues unfinished joint episodes exactly.
Phase transition deliberately resets economies/counters/recurrent state; it is a
new phase rather than an exact resume. Historical format-v1/source snapshots are
preserved and incompatible; generate new phase-one checkpoints with current code.
The evaluator now runs both frozen policies from joint snapshots. Baseline
evaluation remains available; see EVALUATION_VALIDATION.md for the new contract.

All 20 tests passed in 290.75 seconds under the required two-core/8-GiB/no-swap
scope, including CPU bitwise joint resume across a reset, four-economy CUDA joint
resume, and existing worker/evaluation regressions. Ruff and ty passed. Planner
artifacts are in `results/planner-final-20260917/` (194 MiB). The test scope is
inactive/dead; no training/test workers remain. Final resources: 59 GiB disk free,
53 GiB available RAM, GPU 0% / 34 MiB.

## Earlier milestones and historical validation

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
- Training supports free-market, fixed US-federal and joint learned taxes. Held-out
  CPU evaluation supports all three regimes. Joint evaluation needs its compatible
  phase-one source checkpoint to exclude pretraining seeds. Historical snapshots are intentionally incompatible
  with the current source/version; artifacts are preserved.

## Completed follow-up and next work

`PAPER_RECONCILIATION.md` reconciles the 2020 target with pinned tutorial budgets,
observations and curriculum units, and designs staged runs within the existing
100-update/ten-minute limits. The bounded experiment is now complete; no paper-scale run was launched.
`pilot/evaluate.py` now runs frozen worker and planner policies together, with
fresh episode/recurrent state, full labor cost and full tax support. It records
per-period applied rates, planner returns and common baseline metrics. Tests cover
sampled repeatability, greedy per-bracket actions, frozen weights and pretraining
seed exclusion. Trainer/model/environment/checkpoint source was not changed.

The bounded design has run on CPU: all three regimes × three seeds reached 100 updates,
then completed 90 common-seed held-out episodes. See `BOUNDED_EXPERIMENT.md` for
commands, resource/checkpoint reviews, paired results and curriculum measurements.
Treat rankings as diagnostics; joint seeds 42 and 242 produced zero held-out output.
Before paper reproduction, implement an explicit spatial-planner configuration,
matched baseline phase transitions and a justified curriculum/sample-budget plan.
The current nonspatial planner and tiny budgets remain incompatible with claiming
paper reproduction. Do not silently change mechanics or migrate old snapshots.

See `EVALUATION_VALIDATION.md` for the evaluation contract, commands, compatibility
limits, and validation results. Fixed-tax training explicitly selects the scaled
US-federal model and disables tax annealing. Evaluation uses full labor costs,
new episode seeds, frozen weights, and the checkpoint tax regime. All 16 tests
passed in 162.90 seconds, including evaluation repeatability for both regimes,
fixed-tax resume equality, and the existing CUDA smoke test. Ruff and ty passed.
Evaluation artifacts are in `results/evaluation-final-20260917/` (20 MiB).

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

## Latest validation (2026-09-19)

All 20 CPU tests passed across the full suite and one corrected test-fixture rerun;
two CUDA tests were skipped because GPU availability could not be verified
(`nvidia-smi` could not communicate with the driver). Full run: 258.18 seconds;
targeted greedy test: 14.72 seconds. Ruff lint/format and ty passed. Commands,
coverage, the test-helper correction and retained JSON report paths are recorded
in `EVALUATION_VALIDATION.md`. Both resource-limited scopes are inactive/dead;
no test/training workers remain. Latest free disk: 52 GiB; available RAM: 53 GiB.
Earlier GPU availability observations above are historical.


## Latest bounded experiment (2026-09-19)

See [BOUNDED_EXPERIMENT.md](BOUNDED_EXPERIMENT.md). Artifacts are preserved in
`results/bounded-20260919/` (2.9 GiB). All 45 training segments passed; all nine
final checkpoints were selected and hashed before the 90-episode evaluation.
Two held-out episodes repeated exactly for each regime's seed-42 final policy;
all checkpoint hashes remained unchanged. Full CPU tests: 20 passed, two opt-in
CUDA cases skipped, 315.18 seconds. Ruff/format/ty and diff checks passed.

Initial sandboxed `nvidia-smi` failed, but host-level checks showed an idle RTX
3090. CPU was retained to keep device/restart trajectories consistent. Final host
resources at 18:33:54: 35 GiB disk free, 41 GiB RAM available, GPU 0% / 34 MiB;
no experiment scopes or training/evaluation/test processes remained. These
supersede the historical resource observations above.

Next work before reproduction: spatial planner inputs/model, matched baseline
phase transitions, and explicit curriculum scaling with new compatibility
contracts. Do not silently modify or migrate these snapshots. Current baseline
labor weights are only 0.09–0.15 versus joint 1.0; joint training's discrete tax
support remains 0–10%, whereas evaluation exposed 0–100% rates. No economic
convergence or paper reproduction has been established.
