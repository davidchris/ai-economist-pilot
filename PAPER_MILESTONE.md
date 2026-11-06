# Paper-aligned profile and matched learning diagnostic

Status: **engineering milestone and diagnostic complete; economic convergence failed**.

The next gate below is now met; see [PRETRAINING_MILESTONE.md](PRETRAINING_MILESTONE.md).
Its `time` observation change makes this milestone's checkpoints incompatible.

Date: 2026-11-04. Target: the original 2020 simulation study, not its human
experiments or the expanded 2021/2022 study. Configuration alignment alone is not
replication.

## Implemented contract

`--profile paper2020` explicitly selects:

- Full public planner spatial observations: 25×25 resource/landmark and index
  maps, plus all public flattened fields. Worker observations remain 11×11.
  Separate policy/value towers encode maps using embeddings and two convolutions,
  followed by two dense layers and a 256-unit LSTM. Private skills and labor are
  still excluded by Foundation's observation construction.
- Rollout horizon 200, recurrent sequence length 50, one optimization epoch,
  and default minibatches of 60 sequences (3,000 transitions). Planner entropy is
  constant 0.1; worker entropy remains 0.025. Learning rates and existing PPO
  clipping, GAE, finite-value and replay checks remain in force.
- Effective trading labor 0.05 instead of the tutorial's 0.0525, by configuring
  the auction component's labor parameter. The pinned simulator is unchanged.
- `--tax-anneal-steps 54000000` by default. The tax ceiling starts at 10% and
  reaches 100% after the specified aggregate phase-two environment steps.
  Foundation still owns masks, taxes, transfers and episode-reset annealing.
  The schedule converts global steps to per-replica episode completions:
  `slope = 0.9 × workers × 1000 / duration`, `warmup = -0.1 / slope`.
  It advances only on episode reset, with resolution `workers × 1000` steps.
- Explicit `--labor-warmup-constant` (default 10,000) for Foundation's original
  automatic phase-one labor warmup. Its unit is positive-mean-reward timesteps
  **per economy**, not aggregate steps. This inherited tutorial choice is not
  established as an exact 2020-paper schedule.

`--phase2-from` now supports free-market, fixed-US-tax and learned-tax treatments.
All transfer the same worker model **and Adam state**, start fresh economies and
recurrent states, and use full labor costs. Fixed-US taxes and learned taxes use
the same configured ceiling schedule under the paper profile. Omitted tax regime
still selects learned taxes. A phase-two checkpoint cannot masquerade as a
phase-one source. Both phases must use the same profile.

The original `pilot` profile retains its defaults and 100-total-update cap.
The paper profile can resume beyond 100 total updates, with **at most 100 new
updates per invocation**, at most four processes, and a 999,999-total-update
ceiling. The systemd two-core/8-GiB/no-swap limits, 600-second timeout and 15-GiB
disk reserve remain mandatory. A large requested horizon is not a timeout promise.

Checkpoints use **format 3**. Profile, curriculum parameters and phase provenance
are saved, inherited on resume and checked for drift. Full Foundation/RNG,
optimizer and recurrent snapshots still define exact update-boundary continuation.
Historical format-2 checkpoints and results remain untouched; there is no migration.
The old implementation is preserved in Git commit `77bf2cd`.

Evaluation preserves each checkpoint's spatial contract, freezes policies and
uses full labor costs and full tax support. Evaluating before the training ceiling
reaches 100% remains an out-of-support diagnostic. The first phase-two evaluation
point is exactly at the curriculum endpoint: the 100% action has just become
available at reset and has not yet been sampled during training. The final
checkpoint has a further 100,000 training steps with full support.

## Prespecified diagnostic

Run `pilot/experiment.py` through the existing uv environment. It records every
invocation and resource check, archives source and lockfile provenance, and stops
on the first failed segment. Each training/evaluation child uses a separate
bounded systemd scope. Historical directories are never reused.

| Setting | Diagnostic |
| --- | --- |
| Training seed | 42; four economies seeded 42–45 |
| Phase one | 200,000 environment steps, free market |
| Phase two | 200,000 steps **per** free-market/fixed-US/learned-tax treatment |
| Shared initialization | Exact same phase-one model and Adam checkpoint |
| Horizon / sequence / epochs | 200 / 50 / 1 |
| SGD minibatch | 4 sequences = 200 transitions |
| Labor warmup constant | 100, deliberately compressed |
| Tax ceiling duration | 100,000 aggregate phase-two steps, deliberately compressed |
| Segment | 25 updates = 20,000 environment steps |
| Evaluation checkpoints | 100,000 and 200,000 steps in each phase/treatment |
| Held-out seeds | 10000–10009, sampled actions, full 1,000-step episodes |

Total training is 800,000 environment steps (3.2M worker transitions), including
200,000 planner transitions. Evaluation is 80,000 separate environment steps.
One training seed cannot estimate training-seed variability or establish a stable
ranking. The two evaluation points diagnose progress, not statistical convergence.
Both phase-two checkpoints permit full tax support; only the final checkpoint has
trained with that full support. All three regimes have matched pretraining,
adaptation budgets and labor cost.

With four rather than sixty economies, each update collects 800 instead of
12,000 environment steps. Reducing the diagnostic minibatch to 200 preserves
16 worker / 4 planner optimizer steps per iteration, but changes gradient noise
and updates per environment step. This is an explicit reduced-scale experiment.
Default paper-profile minibatches remain 3,000; they cannot recover the paper's
batch size under four replicas and a 200-step horizon.

The original reference is [Appendix B and section 4.2 of the 2020 paper](https://arxiv.org/html/2004.13332v1#A2).
Remaining replication work includes a validated Saez comparison, paper-scale
collection/budgets, ten training seeds, convergence-window analysis, and auditing
remaining optimizer/initialization and environment-version differences. This
PyTorch port is not an exact TensorFlow implementation or checkpoint conversion.

## Validation and measured results

Ruff lint, Ruff format verification, ty and `git diff --check` passed. The five
new CPU acceptance cases passed in **258.82 s**, covering full spatial planner
inputs and gradients, replica-count-invariant tax endpoints, exact restart across
an episode reset for all three matched treatments, repeatable evaluation,
curriculum/profile drift rejection, and continuation from update 100 to 102.
The first run found a test-helper issue comparing scalar NaNs in Foundation's
undefined no-trade diagnostics. The helper now compares their bit representations;
no simulator, reward or numerical training guard was relaxed.

Commands (each pytest invocation inside a separate systemd scope):

```sh
uv run --no-sync ruff check pilot
uv run --no-sync ruff format --check pilot
uv run --no-sync ty check
git diff --check
systemd-run --user --scope --unit=ai-econ-paper-unit2-20261104 \
  -p CPUQuota=200% -p MemoryMax=8G -p MemorySwapMax=0 \
  timeout --signal=TERM --kill-after=10s 600s \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 uv run --no-sync pytest -x -v pilot/test_paper.py
systemd-run --user --scope --unit=ai-econ-paper-regression-20261104 \
  -p CPUQuota=200% -p MemoryMax=8G -p MemorySwapMax=0 \
  timeout --signal=TERM --kill-after=10s 600s \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 RUN_CUDA_RESUME_TEST=1 \
  uv run --no-sync pytest -x -v \
  -k 'not paper_matched and not paper_continues and not spatial_planner_and'
```

The first command collected five cases before the separate opt-in CUDA acceptance
case was added; the second selection includes that new case and the entire prior
suite. Together they cover all current tests. Live scope properties were verified
as `CPUQuotaPerSecUSec=2s`, `MemoryMax=8589934592`, `MemorySwapMax=0`.
Pre-regression resources: 243 GiB disk free, 43 GiB available RAM, load
1.14/1.16/0.90, RTX 3090 idle at 34 MiB. These are observations, not reservations.

The regression/CUDA selection passed **23 tests in 402.07 s**. Combined with
the five new CPU cases, **all 28 tests passed**, including the three opt-in CUDA
cases (worker, nonspatial joint, spatial joint resume/CPU portability). The live
regression scope's observed memory peak was 4,274,962,432 bytes (3.98 GiB), below
its 8-GiB limit.

The diagnostic was launched with:

```sh
uv run --no-sync python pilot/experiment.py \
  --output results/paper-medium-20261104 --device cuda
```

All generated artifacts are ignored under `results/paper-medium-20261104/`;
historical results remain untouched.

### Completed training and state audit

All 40 segments passed: 800,000 environment steps, 3.2M worker transitions,
200,000 planner transitions and 800 completed training episodes. All models
changed parameters in every applicable segment. No action-mask, nonfinite-value,
recurrent-replay, checkpoint or compatibility check failed.

| Phase/treatment | Steps | Training seconds | Steps/s | Peak sampled RSS (GiB) | Final checkpoint (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shared free-market pretraining | 200,000 | 739.357 | 270.505 | 4.016 | 19.942 |
| Free-market phase two | 200,000 | 733.636 | 272.615 | 4.003 | 19.943 |
| Fixed-US-tax phase two | 200,000 | 740.001 | 270.270 | 4.000 | 19.945 |
| Learned-tax phase two | 200,000 | 892.065 | 224.199 | 4.074 | 40.210 |

Training seconds are sums of per-invocation loop times, not sums of inherited
cumulative rollout/update timers. Summed process RSS can double-count shared
pages. Longest outer training invocation: **104.18 s** against a 600-second cap.
Largest PyTorch GPU tensor allocation: **163.59 MiB**. Maximum worker recurrent
replay error was `2.86e-6`; maximum planner error was `3.81e-6` (limit `1e-4`).
Every final segment checkpoint passed CPU/CUDA forward agreement and CPU loading.
These throughput measurements include mostly inactive economies; they are not
runtime guarantees for productive economies, larger batches or paper-scale runs.

The separate bounded `pilot/audit.py` invocation passed in 4.3 s. All selected
phase-two economies have labor weight exactly 1.0; fixed-US and learned-tax
ceilings are exactly 1.0 at both selected checkpoints. Free-market taxes remain
disabled. Final pretraining labor weights span 0.992917–0.996849.
Planner layout is `flat=[119]`, `world-map=[6,25,25]`,
`world-idx_map=[2,25,25]`, `action_mask=[154]`. The eight additional flat features
relative to the nonspatial layout are the four workers' public row/column locations.

Shared worker pretraining lost productive activity: mean episode productivity in
successive 20,000-step blocks was 276.675, 26.349, 19.223, 21.648, 17.744, 8.140,
0.663, 0.567, 0 and 0 coins. This decline occurs **before the tax planner exists**.
The compressed curriculum and reduced batches are plausible contributors, but
this experiment does not isolate their causal effects.

### Held-out results

All 40 two-episode evaluation invocations passed (80,000 environment steps).
Checkpoints were selected and hashed before any held-out evaluation. All eight
hashes remained unchanged afterward. Means and sample SD below are across ten
episode seeds for **one** training seed, not across independently trained policies.

| Policy | Phase steps | Productivity mean ± SD | Welfare mean ± SD | Mean worker return ± SD |
| --- | ---: | ---: | ---: | ---: |
| Pretraining | 100,000 | 26.0776 ± 29.2484 | 1.9043 ± 2.7019 | −2.4517 ± 2.6284 |
| Pretraining | 200,000 | 0 ± 0 | 0 ± 0 | −0.9375 ± 0.2449 |
| Free-market phase two | 100,000 | 0 ± 0 | 0 ± 0 | −0.6628 ± 0.2242 |
| Free-market phase two | 200,000 | 0 ± 0 | 0 ± 0 | −0.6615 ± 0.2244 |
| Fixed-US-tax phase two | 100,000 | 0 ± 0 | 0 ± 0 | −0.6953 ± 0.2178 |
| Fixed-US-tax phase two | 200,000 | 0 ± 0 | 0 ± 0 | −0.7635 ± 0.2942 |
| Learned-tax phase two | 100,000 | 0 ± 0 | 0 ± 0 | −0.6390 ± 0.1696 |
| Learned-tax phase two | 200,000 | 0 ± 0 | 0 ± 0 | −1.1943 ± 0.2971 |

All phase-two held-out episodes had equality 1 and zero taxes collected: equality
at zero output, not successful redistribution. Paired welfare and productivity
differences are all zero. This supplies no useful tax-policy ranking or evidence
that the paper's economic finding has been reproduced. Common episode seeds do
not guarantee identical future random shocks after different actions.

The 100k pretraining checkpoint retained some productive behavior, while the
prespecified 200k source checkpoint had lost it. We did not select an earlier
checkpoint after seeing these results or change the experiment's training inputs.
The failure predates planner training and persists under every matched treatment.
It does not establish whether curriculum compression, batch size, optimization,
sample budget or their interaction caused the failure.

**Next gate:** establish sustained productive worker behavior before starting
another planner comparison. Test a less compressed phase-one curriculum and a
justified batch/sample budget with explicit production and exploration diagnostics.
Use separate development and final evaluation seeds if tuning these choices.
Do not simply extend the current inactive-policy checkpoints and call that paper
replication. Saez and multi-training-seed comparisons remain subsequent work.

### Preserved artifacts and final state

Under `results/paper-medium-20261104/` (about 1 GiB):

- `protocol.json`, `commands.jsonl`, per-invocation resource/exit records and logs.
- `source/`, `provenance.json`, `implementation.patch`: the exact Python source,
  dependency lock and source hashes; source remained unchanged during execution.
- Forty unique training directories and their checkpoints/committed logs.
- `selected-checkpoints.json`, `curriculum-audit.json` and forty evaluation directories.
- `evaluation-records.csv` (80 rows), `aggregates.json`,
  `paired-differences.json` (40 matched episode contrasts), and
  `training-curves.json` (40 blocks of 20 completed training episodes).
- `pretraining-output.html`: the observed productivity decline during pretraining.

Final Ruff lint/format, ty and `git diff --check` passed. At 13:02 local time:
241 GiB disk free, 43 GiB available RAM, load 2.67/3.66/2.95, RTX 3090 idle at
34 MiB. No experiment/test scopes or matching trainer/evaluator/orchestrator/audit
processes remained. No upstream source, historical result, other service or disk
configuration was changed.
