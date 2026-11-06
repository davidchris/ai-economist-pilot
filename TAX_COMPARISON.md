# Matched tax comparison (2026-11-05/06)

**The learned-tax arm collapsed on all three seeds, and the planner was close to random
throughout.** This does not show that learned taxes are worse than the alternatives. It
shows that this planner did not learn within 4.4M steps, while its near-random rates grew
with the ceiling until workers stopped producing. The fixed-tax and free-market arms stayed
productive. This is no convergence or paper reproduction claim.

## Protocol

`pilot/compare.py` continues each gated pretraining checkpoint (seeds 42, 142, 242) through
the real phase transition, for 4.4M steps per treatment, with `--tax-anneal-steps 3600000`.
Both tax treatments ramp under that ceiling in training. Fixed-tax rates start at a flat
10% and reach the full US brackets about 1.08M steps in. Policies were evaluated with
sampled actions on common held-out seeds 40000–40009 at 3.6M and 4.4M, with the ceiling
removed and full labor cost. The protocol and welfare as the primary metric were fixed
before launch. Each arm is checked with the prespecified pretraining gate. Its no-op
reference comes from the free market, so it is only approximate for the tax treatments.

## Results

Means of the three training seeds. Paired by episode seed within each training seed.

| Treatment | Steps | Output | Welfare | Worker return | Taxes collected | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Free-market | 3.6M | 2,070 | 231.5 | 41.7 | 0 | |
| Fixed-tax | 3.6M | 1,839 | 212.2 | 50.8 | 357.6 | |
| Learned-tax | 3.6M | 65 | 13.6 | 0.8 | 46.6 | |
| Free-market | 4.4M | 2,026 | 238.4 | 44.2 | 0 | 3/3 pass |
| Fixed-tax | 4.4M | 1,525 | 172.5 | 45.5 | 284.1 | 3/3 pass |
| Learned-tax | 4.4M | 13 | 2.1 | −2.2 | 9.6 | 0/3 pass |

| Welfare contrast at 4.4M | Mean | 95% interval, n=3 seeds | By seed (42, 142, 242) |
| --- | ---: | ---: | ---: |
| Learned-tax minus free-market | −236.3 | −282.1 to −190.4 | −219.4, −256.0, −233.4 |
| Learned-tax minus fixed-tax | −170.4 | −227.0 to −113.8 | −147.3, −192.8, −171.1 |
| Fixed-tax minus free-market | −65.9 | −79.5 to −52.3 | −72.2, −63.2, −62.3 |

At 3.6M, fixed-tax minus free-market welfare was −19.4 (−69.0 to 30.2), with mixed signs.
That contrast is not stable between the two evaluation points. With three seeds, the
intervals are descriptive. Fixed-tax output also fell from 3.6M to 4.4M on every seed
(1,565→1,467, 2,013→1,666, 1,938→1,443), and seed 242's free-market output fell from 2,129
to 1,750, so these policies are not stationary at this budget.

Seeds 142 and 242 ended exactly at the no-op policy: output 0, about 100% no-ops, return
−1.0. Seed 42 ended at output 39 with return −4.5, below the no-op return of −0.97.

## Why the learned-tax arm collapsed

Seed 42 training episodes:

| Ceiling | Effective rate | Output | Worker return | Planner return |
| ---: | ---: | ---: | ---: | ---: |
| 12% | 4.6% | 1,685 | 27.8 | 187 |
| 42% | 17% | 1,568 | 36.9 | 180 |
| 52% | 26% | 1,409 | 36.1 | 189 |
| 67% | 30% | 873 | 23.1 | 122 |
| 77% | 27% | 487 | 11.0 | 79 |
| 82% | 23% | 184 | 1.3 | 35 |

Measured: the planner's own return fell with output, so it did not gain from this.

| Planner measure, tax-decision steps | Seed 42, 3.6M | Seed 42, 4.4M | Seed 142, 4.4M | Seed 242, 4.4M |
| --- | ---: | ---: | ---: | ---: |
| Entropy, share of maximum | 0.90 | 0.90 | 0.82 | 0.76 |
| Mean top-action probability (uniform 0.045) | 0.16 | 0.15 | 0.25 | 0.28 |
| Spread of chosen rates, SD (uniform grid 0.303) | 0.297 | 0.314 | 0.315 | 0.359 |
| Mean chosen rate | 0.61 | 0.64 | 0.38 | 0.42 |

Median planner approximate KL per update, seed 42, is 4.7e-7, averaged over mostly masked
steps. Fixed-tax rates stop at 37%; learned rates often exceed that in low brackets, where
every worker pays.

**Interpretation, not established:** workers face near-random, often high, unpredictable
rates as the ceiling rises, so doing nothing becomes their best response, and the planner
gets no useful signal once production stops. Possible reasons the planner stays near
uniform include its entropy coefficient (0.1, paper Table 2) against small rewards, its one
decision per 100 steps, the low 1e-4 learning rate, and a budget of 1% of the paper's 400M
steps. None of these was tested. A planner-free control with uniform random rates would
test the first half of this explanation directly.

## Deviations and run events

- **Critic check trips: 14 of 330 learned-tax segments, 0 of 660 others.** All 14 were
  continued from the published checkpoint. Replay of all checks on the 14 checkpoints:
  worker policy at most 2.9% of tolerance, planner policy 0.3%, planner value 5.5%. Worker
  value failed again on 13 of 14 and passed on one, at up to 181% of tolerance and
  2.24e-5. This is far more than the "about 2" estimated in advance, because collapsed
  arms have critic values near zero.
- **Approval changed during the run.** Approval was first per segment, after review. Trips
  became frequent enough to stop an arm for most of a leg, so `compare.py` was changed to
  accept named future segments of arms already seen collapsing: seed 142 from update 4200,
  seed 242 from 3450. Only the critic check qualifies, and `train.py` is unchanged. Seed
  242's learned-tax segments before 3800 and all free-market and fixed-tax segments had no
  trips. `approved-critic-failures.jsonl` lists each approval.
- **Driver bug.** That change let approved segments ignore the pause. One leg reached the
  two-hour limit and was killed during `learned-tax-s242-005250`, which was retired and
  rerun from its parent. Both faults are fixed and tested.
- **Lost log.** Leg 6 retried `learned-tax-s42-005200` before its failure was read, which
  overwrote the first log. The first attempt's directory is preserved and the failure
  repeated. Its checkpoint bytes differ from the retry's; only the check failure was
  compared, so this does not verify exact restart.
- **NVIDIA packages changed at 16:44 on 2026-11-05, not by this work.** The packages went
  from 610.57 to 615.71 while the kernel module stayed at 610.57, so `nvidia-smi` fails until
  a reboot. CUDA kept working. Seed 42's free-market and fixed-tax arms finished before the
  change, seed 42's learned-tax and seed 142's arms span it, and seed 242's ran after it.
  Seed 42's collapse began before it, at output 245 by 16:11.
- `source/` in the run directory is the source at launch.
  `compare-amended-20261106.py` is the final driver. `diagnostics/` holds the replay and
  planner scripts behind this document, which are not tracked.

## Reproduction

Each child runs in its own scope (CPUQuota=200%, MemoryMax=8G, MemorySwapMax=0, 600 s,
single-thread BLAS/OpenMP). Up to three ran concurrently, so up to six cores in total.

```sh
# Rerun until COMPLETE. Read each leg's result before starting the next.
{ seq -f 'learned-tax-s142-%06g' 4200 50 5500; seq -f 'learned-tax-s242-%06g' 3450 50 5500; } |
  xargs uv run --no-sync python pilot/compare.py --output results/compare-20261105 \
    --max-seconds 6000 --accept-critic-failure learned-tax-s42-005200
# Results: report.json, report-<treatment>-s<seed>.json
```

## Measurements and validation

| Measure | Value |
| --- | ---: |
| Elapsed, 12 legs | about 19.4 h |
| Training segments with checkpoints | 990 (976 fully passed, 14 accepted) |
| Longest training segment, limit 600 s | 257.7 s |
| Throughput, free-market and fixed-tax | 266–360 steps/s |
| Throughput, learned-tax | 169–222 steps/s |
| Peak sampled process RSS, can double-count | 4.78 GiB |
| Peak PyTorch GPU allocation | 162 MiB |
| Largest checkpoint | 40.2 MiB |
| Results size | 37 GiB |

On the final source, `uv run --no-sync pytest pilot/test_compare.py pilot/test_pretrain.py`
passed 18 tests in a resource-limited scope (3.46 s), as did Ruff lint and format, ty and
`git diff --check`. The full suite was not rerun: `pilot/train.py`, `environment.py`,
`model.py` and `checkpoint.py` are unchanged since it last passed. The driver paths were
exercised by a one-seed, 80k smoke run and by this run. Planner measures use two episodes
per checkpoint (seeds 50000–50001) and the evaluation episodes above. At 06:33 local time on
2026-11-06: 181 GiB disk free, 41 GiB RAM available, no task processes or scopes remaining.

## Next work

1. Find out why the planner does not learn before spending on any further comparison:
   a uniform-random planner control, then planner settings at a fixed moderate ceiling.
2. Any change to `pilot/train.py` invalidates the pretraining checkpoints, so make the
   scaled critic tolerance part of that change.
3. Keep treating fixed-tax against free-market as provisional, since its welfare contrast
   moved from −19.4 to −65.9 between evaluation points.
