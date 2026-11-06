# Recreate the AI Economist locally

Research date: 2026-09-15.

Current milestone (2026-11-05): [productive worker pretraining](PRETRAINING_MILESTONE.md).
Latest result (2026-11-06): [matched tax comparison](TAX_COMPARISON.md); the planner did not learn.
Three training seeds sustain production at full labor cost on fresh episode seeds and
through a full-cost phase-two hold. No tax comparison has been run on them, and
economic convergence remains unestablished.

Previous milestone (2026-11-04): an explicit `paper2020` profile adds spatial planner
inputs, matched phase-two baseline transitions, and explicit tax-curriculum units.
See [the paper-profile milestone](PAPER_MILESTONE.md) for compatibility, validation
and the matched learning diagnostic. All 28 tests passed. The 800k-step diagnostic
and 80 held-out episodes completed, but workers lost productive behavior during
pretraining and all final treatments produced zero output. Productive worker
pretraining is the next gate; economic convergence remains unestablished.
Historical pilot findings below remain preserved.

Current engineering work (2026-09-17): local Git repository with uv-managed
dependencies, Ruff formatting/linting, ty type checking, and pytest. The resumable
training contract and acceptance commands are in [restart validation](RESUME_VALIDATION.md).
Free-market/fixed-tax training and held-out evaluation are documented in
[evaluation validation](EVALUATION_VALIDATION.md).
Joint worker–planner training and the phase transition are documented in
[planner validation](PLANNER_VALIDATION.md).
See [HANDOFF.md](HANDOFF.md) for remaining work and [UPSTREAM.md](UPSTREAM.md)
for the separately supplied Foundation source dependency.

**The training-server pilot passed on CUDA and CPU.** The RTX 3090 run achieved 189 environment steps/second under a two-core CPU limit. See [pilot results](PILOT_RESULTS.md) for measurements and limitations.

This directory contains papers, the pinned simulator source, pilot code, and results.
A separate Python 3.12/PyTorch environment is installed on the training server at the repository root.
The sections below describe the historical reproduction requirements. The [pilot instructions](pilot/README.md) describe the tested modern stack.

## Papers and scope

- **Original study (2020):** [The AI Economist: Improving Equality and Productivity with AI-Driven Tax Policies](papers/ai-economist-2020-2004.13332v1.pdf). This is the four-agent taxation simulation assumed here. [arXiv record](https://arxiv.org/abs/2004.13332)
- **Extended study (2021):** [Optimal Economic Policy Design via Two-level Deep Reinforcement Learning](papers/ai-economist-2021-2108.02755v1.pdf). The authors identify this as a substantial extension. [arXiv record](https://arxiv.org/abs/2108.02755)
- **Journal publication (2022):** *Taxation policy design via two-level deep multiagent reinforcement learning*, Science Advances, DOI `10.1126/sciadv.abk2607`. [Author's publication record](https://parkes.seas.harvard.edu/publications/ai-economist-taxation-policy-design-two-level-deep-multiagent-reinforcement)

PDF provenance and checksums are in [SOURCES.md](SOURCES.md). The 2021 PDF is a preprint, not the journal PDF.

## What needs to run

| Part | Required behavior |
| --- | --- |
| Foundation environment | Four workers occupy a 25 × 25 resource map. An episode lasts 1,000 steps. |
| Economic actions | Workers gather wood and stone, trade through a double auction, and build houses for coins. Skills and labor costs affect returns. |
| Tax system | A planner selects seven marginal tax rates every 100 steps. Collected taxes return equally to workers. |
| Worker learning | A shared policy learns individual behavior from each worker's observations. Rewards reflect utility changes. |
| Planner learning | A separate policy learns taxes that improve the chosen social welfare measure. |

These are reinforcement learning policies, not language models. The original paper describes the environment and recurrent policy networks in Sections 2–3 and Appendices A–B. [Original paper](https://arxiv.org/html/2004.13332v1)

## Available code, models, and data

The [official repository](https://github.com/salesforce/ai-economist) contains the Foundation engine, notebooks, plotting utilities, and training examples. Salesforce archived it on November 6, 2023. The inspected master commit is `a84d5f3fdcabb207d9fde7754d34906903b3e184`. Its [package metadata](https://github.com/salesforce/ai-economist/blob/a84d5f3fdcabb207d9fde7754d34906903b3e184/setup.py) declares version `1.7.1`.

Useful entry points:

- [Basic simulation notebook](https://github.com/salesforce/ai-economist/blob/master/tutorials/economic_simulation_basic.ipynb): environment creation, random actions, and visualization.
- [RLlib notebook](https://github.com/salesforce/ai-economist/blob/master/tutorials/multi_agent_training_with_rllib.ipynb): wrapper, PPO training, and rollout visualization.
- [Two-level training guide](https://github.com/salesforce/ai-economist/blob/master/tutorials/two_level_curriculum_learning_with_rllib.md): curriculum, models, phase configurations, and checkpoints.
- [Training script](https://github.com/salesforce/ai-economist/blob/master/tutorials/rllib/training_script.py): the relevant executable for taxation experiments.

The original artificial economy generates its own experience. No financial feed, Salesforce account, API key, or external economic dataset is necessary. This follows from the [basic notebook's self-contained environment configuration](https://github.com/salesforce/ai-economist/blob/master/tutorials/economic_simulation_basic.ipynb).

For the extended study, the authors also provide:

- [Zenodo code and analysis archive](https://zenodo.org/records/5644182): `code.zip`, `analysis.zip`, and a license, totaling about 3.5 MB.
- [Dryad experimental data](https://doi.org/10.5061/dryad.bnzs7h4c4): a 3.09 GB download, approximately 20 GB expanded. It includes configurations, training histories, final agent/planner models, and episode logs.

Thus, replaying trained policies and reproducing plots can precede retraining. Archive contents and checkpoint loading were not tested here. The published checkpoint inventory concerns the extended study.

## Software and this machine

The [official training guide](https://github.com/salesforce/ai-economist/blob/master/tutorials/two_level_curriculum_learning_with_rllib.md) specifies these legacy versions:

| Dependency | Version |
| --- | --- |
| Python | 3.7 |
| TensorFlow | 1.14 |
| Ray with RLlib | 0.8.4 |
| Gym | 0.21 |
| AI Economist | At least 1.5 in the guide, 1.7.1 in inspected master |

The [package requirements](https://github.com/salesforce/ai-economist/blob/master/requirements.txt) also pin NumPy 1.21.0, SciPy 1.6.3, pandas 1.2.4, matplotlib 3.2.1, and PyYAML 5.4.1. That file omits TensorFlow, Ray, and Gym. Installing the package alone does not install the training stack.

A local Apple silicon Mac was also inspected, with Python 3.14 as its default interpreter.

**Python 3.14 and native Apple silicon are incompatible with the published TensorFlow 1.14 wheel set.** Its published builds include Python 3.7 and Intel macOS/Linux, with no arm64 build. [TensorFlow release files](https://pypi.org/project/tensorflow/1.14.0/#files)

Historical reproduction options:

1. Use an isolated Linux x86_64 environment with Python 3.7 for the historical stack.
2. On this Mac, evaluate x86_64 emulation with a CPU import test before longer runs.
3. If emulation fails or performs poorly, use an Intel/AMD Linux machine or port the training code.
4. For a native port, preserve observation handling, action masks, recurrent state, rewards, and curriculum behavior.

These are implementation recommendations, not verified installation results. Updating Ray or TensorFlow requires code changes: the [training script](https://github.com/salesforce/ai-economist/blob/master/tutorials/rllib/training_script.py) uses legacy `PPOTrainer` APIs. Gym 0.21 also has [reported installation failures](https://github.com/openai/gym/issues/3200). A reproducible environment needs validated packaging tools and transitive dependency pins.

### CPU versus GPU

The later [training guide](https://github.com/salesforce/ai-economist/blob/master/tutorials/two_level_curriculum_learning_with_rllib.md) reports 16 CPUs and 60 GB RAM. Its supplied configurations use zero GPUs. This Mac has adequate memory relative to that example, but runtime compatibility remains unresolved.

The original paper reports 15 CPUs and two GPUs. It uses 60 parallel environments, 50 million phase-one steps, and 400 million phase-two steps. [Appendix B](https://arxiv.org/html/2004.13332v1#A2)

The [training-server pilot](PILOT_RESULTS.md) provides initial throughput and memory measurements.
Full retraining still needs a validated planner and measurements at a larger worker count.
The Apple GPU does not provide the historical CUDA execution path. Salesforce's [WarpDrive notebook](https://github.com/salesforce/ai-economist/blob/master/tutorials/multi_agent_gpu_training_with_warp_drive.ipynb) demonstrates the COVID-19 scenario, so it is not the direct taxation reproduction route.

## Reproduction milestones

### 1. Run and inspect one episode

After the isolated runtime is ready, obtain a pinned source checkout:

```sh
git clone https://github.com/salesforce/ai-economist.git upstream
cd upstream
git checkout a84d5f3fdcabb207d9fde7754d34906903b3e184
```

Use the basic notebook to create the environment, reset it, and sample permitted actions. Run one full episode. Verify that resources, coins, taxes, and episode termination behave correctly. Save a plot and episode log.

Success means that the environment runs. Random workers do not demonstrate learned economic behavior.

### 2. Train workers under a fixed tax policy

Start with the RLlib notebook. Use a small explicit worker count and a short training budget. Verify finite rewards, policy updates, checkpoint restoration, and evaluation rollouts.

Train the free-market case first with `disable_taxes: true`. For the fixed baseline, enable taxes and set `tax_model: us-federal-single-filer-2018-scaled`. Keep planner learning disabled. Preserve the same environment and evaluation procedure across policies. The [tax component](https://github.com/salesforce/ai-economist/blob/master/ai_economist/foundation/components/redistribution.py) defaults to `model_wrapper`, so US-federal bracket spacing alone does not select fixed tax rates.

### 3. Train workers and planner together

Use the provided [phase-one configuration](https://github.com/salesforce/ai-economist/blob/master/tutorials/rllib/phase1/config.yaml) to train workers without taxes. It includes gradual labor costs.

For phase two, set `general.restore_tf_weights_agents` to the actual phase-one checkpoint path. Enable planner training and preserve tax and entropy curricula. The [script](https://github.com/salesforce/ai-economist/blob/master/tutorials/rllib/training_script.py) reads this top-level `general` field, despite different wording in the tutorial.

Once configurations and dependencies pass the preceding checks, the upstream entry points are:

```sh
cd tutorials/rllib
python training_script.py --run-dir phase1
# Set the phase-two checkpoint path before the next command.
python training_script.py --run-dir phase2
```

These commands were inspected against source, not executed.

### 4. Reproduce a specified paper result

**Choose the target paper before adopting the default budgets.** The later [phase-one](https://github.com/salesforce/ai-economist/blob/master/tutorials/rllib/phase1/config.yaml) and [phase-two](https://github.com/salesforce/ai-economist/blob/master/tutorials/rllib/phase2/config.yaml) configurations differ from the original experiment:

| Setting | Original 2020 | Supplied later configurations |
| --- | --- | --- |
| Phase-one steps | 50 million | 25 million |
| Phase-two steps | 400 million | 4 billion |
| Planner spatial observations | Enabled | Disabled |
| Recurrent sequence length | 50 | 25 |

The later step budgets follow from `episodes × 1,000`. [Original Appendix B](https://arxiv.org/html/2004.13332v1#A2)

The saved 2021 preprint differs again: Table 3 specifies 25 million phase-one steps and 1 billion phase-two steps. Its training-strategy prose specifies 50 million phase-one steps. These discrepancies require an explicit choice of reference configuration. [Extended preprint, Methods](https://arxiv.org/pdf/2108.02755v1)

For the original result, compare free market, US-federal, Saez, and learned taxes across ten seeds. Measure productivity, equality, welfare, and tax schedules. The reported gain is 16% over Saez for the equality-productivity objective. [Original results](https://arxiv.org/html/2004.13332v1#S4.S3)

For the extended study, use its archived experiment configurations and analysis code. Verify metric definitions, map layout, skill assignment, policy observations, and seed handling. Record environment steps separately from per-agent transitions. Preserve the source revision, dependency lock, configurations, checkpoints, evaluation logs, and comparison plots.

**Budget and observation reconciliation is complete:** see
[PAPER_RECONCILIATION.md](PAPER_RECONCILIATION.md) for differences and a bounded longer-run design.
The separate planner policy and phase transition are implemented; see
[planner validation](PLANNER_VALIDATION.md).
Resumable training and [held-out worker and joint evaluation](EVALUATION_VALIDATION.md) are implemented.
The [collection profiling comparison](COLLECTION_PROFILE.md) is complete.


The [bounded three-seed experiment](BOUNDED_EXPERIMENT.md) is complete: all three
regimes reached 100 updates per seed, followed by 90 common-seed held-out episodes.
Resource/checkpoint gates and repeatability checks passed. Results show large
training-seed variation and zero output for two joint policies; they validate the
workflow, not convergence. Spatial planner observations, matched baseline phase
transitions and curriculum scaling remain prerequisites for paper reproduction.
