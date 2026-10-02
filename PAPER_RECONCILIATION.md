# 2020 target and bounded experiment design

Reconciled 2026-09-19 against [the 2020 paper, sections 4.2–4.3 and Appendices A–B](https://arxiv.org/html/2004.13332v1),
not the expanded 2021 study. This is a configuration audit, not a reproduction.

## Paper target

The paper specifies 50M phase-one and 400M phase-two environment steps;
60 replicas × 200 steps yield 12,000 environment steps per iteration,
48,000 worker transitions and 12,000 planner transitions. It uses sequence
length 50, minibatches of 3,000, and 16 worker/4 planner updates per iteration.
Episodes last 1,000 steps. Results span ten training seeds and the final 20M steps.
Tax limits rise from 10% to 100% over 54M steps; Table 2 lists planner entropy 0.1.
Workers see 11×11 local maps, their own public/private state, market and tax
information. The planner sees the full 25×25 world and public worker state,
but not private skill/labor. Tax decisions occur every 100 steps; the recurrent
planner observes intervening steps with masked no-ops. Table 3 lists trading
labor 0.05. These are the reference targets, not descriptions of this pilot.

## Pinned tutorial and local differences

Audit sources in the preserved revision `a84d5f3fdcabb207d9fde7754d34906903b3e184`:
`upstream/tutorials/rllib/{phase1,phase2}/config.yaml`,
`training_script.py` (termination uses `episodes_total`),
`foundation/scenarios/simple_wood_and_stone/layout_from_file.py`,
`foundation/components/{redistribution,utils}.py`, and local `pilot/{environment,model,train}.py`.

| Setting | Pinned tutorial | Local trainer |
| --- | --- | --- |
| Phase budgets | 25,000 / 4,000,000 episodes = 25M / 4B environment steps, nominally | At most 100 total updates per phase |
| Parallelism | 15 workers × 2 economies = 30 economies | 1–4 economies |
| Rollout / recurrent sequence | 200 / 25 | Defaults 128 / 16 |
| Train batch / SGD minibatch | 6,000 environment steps / 1,500 transitions | `workers × horizon` environment steps / `minibatch_sequences × sequence_length` transitions |
| Epochs | 1 | Default 2 |
| Planner observation | `planner_gets_spatial_info: false` | Same; sorted concatenation of all non-mask planner fields, including appended public per-worker fields |
| Planner architecture | Generic convolution/LSTM wrapper, conditioned on available inputs | Dense 256-unit towers/LSTMs; no map encoder |
| Worker observation | 11×11 maps plus flattened features and masks | Same, four workers share weights; separate recurrent states |
| Planner entropy | 2.0 → 0.125 over 50M global steps | Same numbers, explicitly aggregate phase-two environment steps |
| Tax annealing | `[-100, 0.001]`, per-replica episode completions | Foundation owns this unchanged |
| Labor curriculum | Phase one automatic, constant 10,000; phase two full cost | Same; fixed-tax baseline currently starts from scratch with labor warmup |
| Effective trading labor | `0.25 × 0.21 = 0.0525` at full cost | Same |

Spatial input is a substantive compatibility gap: with the flag false,
`layout_from_file.generate_observations` supplies worker inventories to the planner
but omits both world maps and worker locations. Packing the existing fields cannot
recover missing information. Adding spatial observations later requires an explicit
configuration/model change and newly trained checkpoints. Evaluation preserves the
trained observation contract; it does not retrofit paper observations.

The local tax ceiling is `min(1, 0.001 × (completions + 100))`: it reaches one at
900 completed episodes per economy. Thus four synchronized economies require
3.6M aggregate steps, versus 27M for the tutorial's 30 economies. The curriculum
is not invariant to process count. Do not silently accelerate or rescale it in
an existing run. Local entropy is independently indexed by aggregate steps.
The local optimizer implementation/initialization and phase transition (including
Adam-state transfer) also preclude a claim of historical TensorFlow equivalence.

## Bounded longer experiments

Keep current mechanics, source fingerprints, 100-update cap and exact restart
semantics. No paper-scale run is launched by this implementation. Use four
processes only if the live resource check permits; use CPU if CUDA is unavailable.
Every invocation must use a separate two-core/8-GiB/no-swap systemd scope,
600-second timeout, single-thread BLAS/OpenMP and at least 15 GiB free.

1. Use training seeds 42, 142, 242, separately. Start with four economies,
   horizon 128, sequence 16, two epochs and checkpoint interval five/retention two.
   First target 20 updates = 10,240 environment steps, 40,960 worker transitions.
2. After checking elapsed time, memory, numerical checks and checkpoint size,
   resume to targets 40, 60, 80 and 100 in separate bounded invocations. Each
   adds at most 10,240 environment steps. Total per phase/seed is 51,200 steps,
   204,800 worker transitions and 48 complete episodes plus partial episodes.
   Preserve parent logs/checkpoints. Retention is per directory, not global.
3. Start joint phase two from each corresponding free-market checkpoint;
   use the same staged totals. Independently train the fixed-US baseline with
   the same per-phase 51,200-step ceiling. Report joint pretraining separately:
   its combined worker budget is 102,400 environment steps. A free-market
   continuation or fixed-tax warm-start at full labor cost would require a new
   transition feature before claiming matched phase-two treatment comparisons.
4. Evaluate each final policy on common seeds 10000–10009, two episodes per
   bounded invocation. Default to sampled actions for both policies; report
   greedy results separately. Evaluation rejects both phase-one and phase-two
   training seeds. Select checkpoints before inspecting these held-out results.
5. Report paired per-seed differences in welfare, productivity, equality and
   taxes, plus worker returns and learned-policy tax histories. Keep variation
   across evaluation episodes separate from variation across three training
   seeds. Current output supplies per-seed records and within-policy mean,
   sample SD and SEM; it is not an automatic significance test.

Historical four-process throughput (262.4 environment steps/s) implies about
39 seconds of training-loop time per 20-update segment. This is planning
arithmetic, not a joint-policy measurement or timeout guarantee. Startup,
checkpointing, CPU-only execution and shared load add cost. If the first segment
approaches the timeout, lower the next additional-update count without changing
its saved rollout settings. Never extend the ten-minute scope to compensate.

Acceptance gates: no invalid actions/nonfinite values/replay failures; source and
checkpoint stay compatible; live memory/disk limits hold; restart tests pass;
evaluation is repeatable and frozen. Inspect labor weights and tax-rate support:
these tiny budgets barely advance either curriculum. Short-run rankings are
engineering diagnostics only. Saez, paper spatial observations, matched baseline
phase transitions, curriculum rescaling and larger budgets remain separate work.

Example (substitute a new scope name for each invocation):

```sh
# Before every experiment: df -h .; free -h; uptime; nvidia-smi
systemd-run --user --scope --unit=ai-econ-bounded-phase1 \
  -p CPUQuota=200% -p MemoryMax=8G -p MemorySwapMax=0 \
  env UV_CACHE_DIR=/tmp/ai-econ-uv-cache bash pilot/run.sh cpu \
  --workers 4 --seed 42 --horizon 128 --sequence-length 16 --epochs 2 \
  --updates 20 --checkpoint-every 5 --keep-checkpoints 2
# Resume with --resume PATH --updates 40; phase two uses --phase2-from PATH.
# Explicit output paths must be new; otherwise a unique directory is created.
```

Observed layouts in the joint CPU acceptance checkpoint: worker `flat=[136]`,
`world-map=[7,11,11]`, `world-idx_map=[2,11,11]`, `action_mask=[50]`;
planner `flat=[111]`, `action_mask=[154]`, with seven subspaces of size 22.
These are checkpoint layout checks, not a semantic equivalence proof for every
flattened feature. The evaluator validates them before the first policy forward.
