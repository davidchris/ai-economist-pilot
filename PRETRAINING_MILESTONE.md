# Productive worker pretraining

Status: **prespecified gate passed on three training seeds; this is not economic
convergence or paper replication.** No tax-policy comparison was run.

Dates: 2026-11-04 to 2026-11-05. Follows [PAPER_MILESTONE.md](PAPER_MILESTONE.md),
whose pretraining collapsed to zero output.

## Recommended configuration

```text
--profile paper2020 --workers 4 --minibatch-sequences 4 --labor-warmup-constant 40000
```

4.4M phase-one environment steps. Use `--phase2-from` on these checkpoints (under
`results/pretrain-stage-b-20261104/`, ignored by Git) for the next comparison:

| Training seed | Checkpoint | SHA-256 prefix |
| ---: | --- | --- |
| 42 | `labor40k-005500/checkpoint-005500.pt` | `2310fd907037edb7` |
| 142 | `labor40k-s142-005500/checkpoint-005500.pt` | `848c51608858b22f` |
| 242 | `labor40k-s242-005500/checkpoint-005500.pt` | `79e390ffa6228092` |

The profile's default constant remains 10,000. Pass 40,000 explicitly.

## Findings

**Cause of the collapse: the labor curriculum.** Foundation's automatic warmup uses
`weight = 1 - exp(-integrator / constant)`. The integrator counts steps with positive
mean worker reward, per economy. It rose about 150 per episode in labor-free runs. A
constant of 100 therefore reached weight ~0.79 within the first episode, while the
policy was near-random and paying about 625 labor per worker. Inactivity then became
the better local choice. The unit also is not replica-invariant: the tutorial's 10,000,
if applied at the paper's 60 economies, gives the same aggregate exposure as 150,000 at
four. Whether 10,000 is the paper's exact schedule remains unestablished.

**Reduced batches did not cause it.** In the labor-free control the smallest batch
learned best. Only 16 optimizer steps per update lifted the critic off its ceiling
within the budget. This reverses the earlier suspicion and is a budget-dependent
result, not a claim about the paper's batch size.

**Input change: `time` is appended, as a duplicate.** Upstream's model feeds every
non-spatial, non-mask key to its dense layers, but `pack()` dropped the separate worker
`time` key. Worker `flat` is now 137 wide instead of 136, which matches upstream's input.

Correction (2026-11-06): this was first recorded as a defect in which workers never saw
`time`. That was wrong. Foundation's flattened vector already contains it: over one
1,000-step episode, `flat[130]` was the only element that equaled `time` at every step.
So the change added no new information, and the comment in `pack()` that calls `time`
the only such key "outside Foundation's flattened vector" is misleading. That comment
stays, because `pilot/environment.py` is part of the source fingerprint and editing it
would reject the current checkpoints.

Effect here was modest (seed 42 held-out productivity at 400k: 1,002 to 1,134, close to
noise), and it was not the fix: constant 100 still collapses with it.

**Critic saturation, a slow start rather than a blocker.** A direct probe of the 4-step
arm at 160k found a constant value (V = 17.31 for every agent after about t = 100). That
is near the limit its output weights allow over LSTM outputs bounded by ±1
(1.55·√128 ≈ 17.5), which is consistent with saturation. The 1-step arm's explained
variance was also ~0. With 16 steps, explained variance reached ~0.65 by 360k steps and
~0.85 later. Initialization parity with Keras (Glorot, forget-gate bias) was not tested.

**A no-op policy does not earn zero.** `move.py` runs collection even for a no-op, so a
resource respawning under a stationary agent forces a gather and charges 1 labor. This
explains the small negative returns of the collapsed policies. The mechanic is
preserved; the gate compares against the measured no-op return.

## Prespecified gate

`pilot/pretrain.py:gate`, fixed before any curriculum result. At **each of the last two
evaluated checkpoints**, with sampled actions and full labor cost over ten episodes:

- Lower 95% bound of mean worker return exceeds the measured no-op return.
- Mean productivity exceeds the uniform-random reference.
- Builds occur.
- The policy was **trained** at labor weight ≥ 0.95, not merely evaluated at full cost.

References (development seeds 20000–20009):

| Policy | Productivity | Mean worker return | Labor per worker | Builds |
| --- | ---: | ---: | ---: | ---: |
| No-op | 0 | −1.18 | 5.6 | 0 |
| Uniform random over valid actions | 419 | −103.98 | 701 | 29.2 |

## Screening results (development seeds 20000–20009)

Stage A, labor-free control (constant 1e12), 400k steps, seed 42, before `time` was appended:

| Minibatch transitions | Optimizer steps per update | Training productivity 40k → 360k | Explained variance at 360k | Held-out productivity at 400k |
| ---: | ---: | --- | ---: | ---: |
| 200 | 16 | 622 → 1,107 | 0.65 | 1,002 ± 39 |
| 800 | 4 | 495 → 717 | −0.002 | 949 ± 56 |
| 3,200 | 1 | 577 → 932 | −0.001 | 440 ± 40 |

Stage A2 repeated this with `time` appended: minibatch 200 reached 1,134 ± 65 (seed 42) and
1,181 ± 62 (seed 142) at 400k; minibatch 800 reached 866 ± 26 (seed 42).

Curricula, minibatch 200, seed 42, all with `time` appended:

| Constant | Steps run | Labor weight reached | Outcome |
| ---: | ---: | ---: | --- |
| 100 | 240k | 1.000 | Collapsed: held-out productivity 53 ± 8, return −5.5, 80% no-ops, labor 64 |
| 10,000 | 560k | ~0.75 | Declining: training productivity 997 → 696, no-op share 0.52 and rising. **Stopped early, see limits** |
| 40,000 | 4.4M | 0.974 | **Passed** |
| 150,000 | 1.6M | 0.315 | Highest output at low weight (1,613 ± 56), held-out return −50.7; cannot reach full cost in this budget |

Only the 40,000 arm was run beyond 1.6M. Its seed 42 held-out return at full cost rose
−78.5, −32.2, −23.1, −8.3, −3.6, +8.3, +22.9, +21.0, +22.5, +34.8, +26.4 at each 400k.

Development gate, last two checkpoints (4.0M and 4.4M), mean ± SEM:

| Seed | Productivity | Return | Lower 95% bound of return | Trained weight |
| ---: | --- | --- | --- | --- |
| 42 | 1,881 / 1,870 | 34.8 / 26.4 | 31.1 / 20.6 | 0.961 / 0.974 |
| 142 | 1,453 / 1,550 | 22.4 / 31.7 | 18.6 / 28.3 | 0.963 / 0.974 |
| 242 | 1,535 / 1,650 | 27.2 / 22.4 | 24.4 / 17.5 | 0.962 / 0.972 |

## Final results (fresh seeds 30000–30009)

The final checkpoint of each seed was selected in advance. These episode seeds were
not used for any choice. "Hold" is 240,000 further steps through the real
`--phase2-from` free-market transition: fresh economies and recurrent state, transferred
model and Adam state, labor weight exactly 1.0. Entries are mean ± SEM across ten
episode seeds for one trained policy.

| Seed | Stage | Productivity | Worker return | Labor per worker |
| ---: | --- | ---: | ---: | ---: |
| 42 | Pretrained | 1,706 ± 62 | 23.4 ± 2.5 | 498 |
| 42 | Hold 120k | 1,602 ± 67 | 29.8 ± 2.2 | 444 |
| 42 | Hold 240k | 1,674 ± 75 | 36.8 ± 2.0 | 426 |
| 142 | Pretrained | 1,605 ± 66 | 31.6 ± 1.6 | 434 |
| 142 | Hold 120k | 1,777 ± 82 | 30.1 ± 1.8 | 483 |
| 142 | Hold 240k | 1,497 ± 78 | 29.6 ± 2.8 | 420 |
| 242 | Pretrained | 1,543 ± 87 | 15.6 ± 3.7 | 510 |
| 242 | Hold 120k | 1,503 ± 71 | 31.6 ± 2.1 | 416 |
| 242 | Hold 240k | 1,728 ± 64 | 33.1 ± 1.7 | 473 |

All three pass every check on fresh seeds and the same gate across the hold. Compared
with uniform random, final hold policies produce about 3.9 times the output with about
two-thirds of the labor. Lowest single-episode productivity at hold 240k: 1,346
(seed 42), 1,145 (seed 142), 1,515 (seed 242).

## Limits

- **Far from an efficient economy.** At hold 240k, 58–61% of actions are no-ops, about
  a quarter are market orders and entropy is ~0.9. Productivity has plateaued only
  within this budget; no convergence is shown, and no comparison to the paper's
  reported productivity was made.
- **4.4M steps is not the paper's budget.** It is 9% of 50M, on 4 rather than 60
  economies. Minibatch 200 means 16 times the paper's optimizer steps per transition.
- **40,000 is empirical.** It was chosen from measured learning speed, on seed 42
  development episodes. The paper-equivalent 150,000 would need more than 10M steps.
- **Comparison arms are single-seed.** Only 40,000 has replicate seeds, so the ranking
  among curricula is a strong diagnostic rather than a statistical estimate. Three
  training seeds also do not replace the paper's ten.
- **The 10,000 arm has no endpoint.** It stopped at update 700 when the CPU/CUDA forward
  agreement check found a 1.16e-5 difference against a 1e-5 tolerance, on one critic value
  near 0.007. The policy check had already passed. This was the only failure in 500
  invocations. The check was **not** relaxed (see below). The arm's decline is observed,
  but its eventual collapse is not.
- **Hold tests free-market continuation only.** Fixed-tax and learned-tax phase twos,
  and Saez, remain untested on these checkpoints.

## Decisions for the tax comparison

### CPU/CUDA tolerance: unchanged for now

The check was replayed in a resource-limited scope on the saved observation of all 438
checkpoints that load on current source, under `results/pretrain-*`. The 30 Stage A
checkpoints are incompatible and were skipped.

| Compared output | Worst largest difference | Worst share of tolerance | Failures |
| --- | ---: | ---: | ---: |
| Action probabilities | 2.56e-6 | 5.7% | 0 |
| Critic values | 2.82e-4 | 108% | 1 |

- The one failure is `labor10k-000700`, and it repeats in a fresh process (1.28e-5). The
  largest critic differences pass because those values are large and the tolerance scales
  with each value. The failing value crossed near zero, where only the 1e-5 absolute
  tolerance applies, while that forward pass's largest value was 19.8.
- Policy total-variation distance between devices never exceeded 2.61e-6. Evaluation
  samples from the policy and does not use the critic.
- `train.py` is part of the source fingerprint. Editing the check now would make the three
  recommended checkpoints unloadable, so the cost is about 4.7 hours of concurrent
  retraining on all three seeds, plus holds and evaluations, or a weaker fingerprint.
- At the observed rate, about 990 invocations would trip about twice. Rerunning a failed
  segment from its parent should reproduce the same checkpoint and failure, so continue
  from the checkpoint it published instead. That path was exercised:
  `train.py --resume .../labor10k-000700/checkpoint-000700.pt --updates 705` passed its
  checks in 25.13 s.
- When `train.py` next has to change for another reason, make the critic tolerance scale
  with the largest value in the forward pass, and leave the policy check as it is.

### Tax ceiling: `--tax-anneal-steps 3600000`

- It produces exactly the pinned tutorial's schedule, `[-100.0, 0.001]` (checked through
  `configuration("learned-tax", "paper2020", 4, True, 3600000)`). The ceiling reaches 100%
  after 900 completed episodes per economy, as the paper's 54M does at 60 economies.
- It is also supported by measured learning speed. The passing 40,000 labor curriculum
  reached weight 0.945–0.949 at 3.6M steps across the three seeds. The 10,000 arm, which
  was declining, was about four times faster. This is an analogy between two different
  curricula, not evidence that workers will adapt to taxes.
- It uses episodes per economy, so it gives 4 economies far fewer samples than the paper's
  60 over the same ramp. Both tax treatments have this ceiling during training. Fixed-tax
  rates are clipped to it, so they start at a flat 10% and reach the full US brackets once
  the ceiling passes 37%, about 1.08M steps in. The free-market treatment has no taxes and
  matches on total steps only. Evaluation removes the ceiling for all of them.
- Planned budget: 4.4M phase-two steps per treatment, the 3.6M ramp plus 800k at full
  support, evaluated at both points. 800k is a resource choice and can be extended
  through resume. From measured throughput, nine runs at three concurrent would take about
  17–18 hours. That is an estimate: learned-tax throughput (224 steps/s) was measured on
  mostly inactive economies.

## Compatibility and code

- Worker `flat` is 137 wide. Earlier checkpoints, including all of
  `results/paper-medium-20261104/`, cannot resume or be evaluated with current source.
  They are preserved, unmigrated. The widths in older documents are historical.
- Format version stays 3; the source fingerprint rejects old checkpoints.
- Foundation mechanics are unchanged. New diagnostics only read existing state:
  per-update action shares and explained variance, and per-episode resources gathered.
- `pilot/pretrain.py` runs resumable arms, references, the gate and finalization.
  `pilot/experiment.py` gained `archive_source` and a unit-prefix parameter.

## Reproduction

Each child runs in its own scope (CPUQuota=200%, MemoryMax=8G, MemorySwapMax=0, 600 s,
single-thread BLAS/OpenMP). Up to three scopes ran concurrently, so up to six cores in
total; each individual run stayed within the two-core limit. A launch guard requires
12 GiB of available RAM.

```sh
uv run --no-sync python pilot/pretrain.py --output results/pretrain-stage-a-20261104 \
  --steps 400000 --eval-every 200000 \
  --arm free-mb200:42:1e12:4 --arm free-mb800:42:1e12:16 --arm free-mb3200:42:1e12:64
uv run --no-sync python pilot/pretrain.py --output results/pretrain-stage-a2-time-20261104 \
  --steps 400000 --eval-every 200000 \
  --arm time-mb200:42:1e12:4 --arm time-mb800:42:1e12:16 --arm time-mb200-s142:142:1e12:4
uv run --no-sync python pilot/pretrain.py --output results/pretrain-stage-b-20261104 \
  --steps 1600000 --eval-every 400000 \
  --arm labor10k:42:10000:4 --arm labor40k:42:40000:4 --arm labor150k:42:150000:4
# Repeat until complete; --max-seconds pauses cleanly between segments.
uv run --no-sync python pilot/pretrain.py --output results/pretrain-stage-b-20261104 \
  --steps 4400000 --eval-every 400000 --max-seconds 6300 \
  --arm labor40k:42:40000:4 --arm labor40k-s142:142:40000:4 --arm labor40k-s242:242:40000:4
uv run --no-sync python pilot/pretrain.py --output results/pretrain-stage-b-20261104 \
  --steps 4400000 --eval-every 400000 --finalize labor40k labor40k-s142 labor40k-s242
uv run --no-sync python pilot/pretrain.py --output results/pretrain-control-labor100-20261105 \
  --steps 240000 --eval-every 120000 --arm labor100:42:100:4
```

Stage A used the source before `time` was appended; its archive and patch are in its
directory. In Stage B, the first invocation failed on the 10,000 arm's check, and
`report.json` reflects only the last completing invocation. Use the `report-*.json` and
`final-*.json` files, which cover each arm.

## Validation and measurements

The full suite passed on the training source: **33 tests** in two scopes, 5 in 278.62 s
and 28 in 436.81 s, including opt-in CUDA cases. `pilot/pretrain.py` later gained
finalization, which has no dedicated test and was exercised by the real runs. Its five
tests passed again on the final source (4.04 s), as did Ruff lint and format, ty and
`git diff --check`.

| Measure (40,000 arms) | Value |
| --- | ---: |
| Throughput per scope, three concurrent | 257–273 environment steps/s |
| Longest training segment (40,000 steps), limit 600 s | 170.0 s |
| Peak sampled process RSS, can double-count shared pages | 4.04 GiB |
| Peak PyTorch GPU allocation | 136 MiB |
| Checkpoint size | 20.0 MiB |
| Maximum recurrent replay error, limit 1e-4 | 4.77e-6 |

Results occupy about 13 GiB in four new ignored directories. No historical result, upstream
source, other service or disk configuration was changed. At 01:44 local time on
2026-11-05: 229 GiB disk free, 43 GiB RAM available, load 2.13/2.09/2.61, RTX 3090 idle
at 34 MiB, and no trainer, evaluator or orchestrator processes remaining.

## Next work

The matched comparison has run; see [TAX_COMPARISON.md](TAX_COMPARISON.md) for its results
and next work. The critic-check estimate above was too low for collapsed arms.
