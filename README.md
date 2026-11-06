# AI Economist pilot

An independent attempt to recreate the simulation study from
[*The AI Economist: Improving Equality and Productivity with AI-Driven Tax Policies*](https://arxiv.org/abs/2004.13332)
(Zheng et al., 2020), using a modern PyTorch PPO trainer on top of Salesforce's original
[Foundation](https://github.com/salesforce/ai-economist) economic simulator.

In the paper, four worker agents gather resources, trade, and build houses in a 25 × 25
world, while a separate planner agent learns tax rates to improve a social welfare
objective. Both learn through reinforcement learning, with the planner reacting to workers
that are adapting to its taxes.

**This project has not reproduced the paper's results.** The training pipeline works and
workers learn to produce at full labor cost, but the learned tax planner does not yet learn.
See [Status](#status).

This project is not affiliated with or endorsed by Salesforce or the paper's authors.

## Status

| Stage | State | Details |
| --- | --- | --- |
| Exact, resumable PPO training | Working | [RESUME_VALIDATION.md](RESUME_VALIDATION.md) |
| Joint worker and planner training | Working | [PLANNER_VALIDATION.md](PLANNER_VALIDATION.md) |
| Held-out evaluation | Working | [EVALUATION_VALIDATION.md](EVALUATION_VALIDATION.md) |
| Paper-aligned configuration | Partial, at about 1% of the paper's phase-two budget | [PAPER_MILESTONE.md](PAPER_MILESTONE.md) |
| Productive worker pretraining | Passed a prespecified gate on three training seeds | [PRETRAINING_MILESTONE.md](PRETRAINING_MILESTONE.md) |
| Learned taxes improve on the baselines | **Not achieved** | [TAX_COMPARISON.md](TAX_COMPARISON.md) |

Latest result, averaged over three training seeds after 4.4M phase-two environment steps,
evaluated on held-out episodes at full labor cost with no tax ceiling:

| Treatment | Output | Welfare | Worker return | Taxes collected |
| --- | ---: | ---: | ---: | ---: |
| Free market | 2,026 | 238.4 | 44.2 | 0 |
| Fixed US-federal tax | 1,525 | 172.5 | 45.5 | 284.1 |
| Learned tax | 13 | 2.1 | −2.2 | 9.6 |

Read this as a failure of this planner to train, not as evidence about learned taxes. The
planner stayed close to random, its rates grew with the tax ceiling, and workers stopped
producing. The paper used 400M phase-two steps across 60 economies and ten seeds; this run
used 4.4M steps across 4 economies and three seeds. Finding out why the planner does not
learn is the next step. The Saez baseline has not been implemented.

## Requirements

- Linux x86_64 with systemd. Every training and test run is confined to a systemd scope.
- [uv](https://docs.astral.sh/uv/) and Python 3.12.
- About 6 GiB for the environment, and 15 GiB free before any run. The lockfile pins a
  CUDA 12.6 build of PyTorch, which is also used for CPU runs.
- A GPU is optional. Tests run on CPU.

## Setup

The simulator is not vendored. Fetch the pinned revision into `upstream/` and apply the one
NumPy compatibility patch, which changes no simulation rules (see [UPSTREAM.md](UPSTREAM.md)):

```sh
git clone https://github.com/davidchris/ai-economist-pilot.git
cd ai-economist-pilot

git init upstream
git -C upstream fetch --depth 1 https://github.com/salesforce/ai-economist.git \
  a84d5f3fdcabb207d9fde7754d34906903b3e184
git -C upstream checkout FETCH_HEAD
sed -i 's/\bnp\.int\b/int/g' \
  upstream/ai_economist/foundation/scenarios/simple_wood_and_stone/layout_from_file.py

uv sync --locked
uv run --no-sync pre-commit install
```

Checkpoints fingerprint the simulator and trainer source, so a different revision or a
changed `pilot/train.py` is rejected on resume by design.

## Usage

Train a small CPU run inside the required resource limits. `pilot/run.sh` adds the
ten-minute timeout and single-thread settings:

```sh
systemd-run --user --scope --unit=ai-econ-example \
  -p CPUQuota=200% -p MemoryMax=8G -p MemorySwapMax=0 \
  bash pilot/run.sh cpu --workers 4 --updates 5 --output results/example
```

Resume with `--resume PATH`, start phase two with `--phase2-from PATH`, and select the
paper-aligned settings with `--profile paper2020`. To evaluate on held-out seeds:

```sh
systemd-run --user --scope --unit=ai-econ-example-eval \
  -p CPUQuota=200% -p MemoryMax=8G -p MemorySwapMax=0 \
  timeout --signal=TERM --kill-after=10s 600s \
  uv run --no-sync python pilot/evaluate.py \
  --checkpoint results/example/checkpoint-000005.pt --seeds 10000 10001
```

Only load checkpoints you trust, because simulator and RNG state are stored as pickle data.

[pilot/README.md](pilot/README.md) covers the trainer in full. `pilot/pretrain.py` and
`pilot/compare.py` drive the multi-seed experiments.

## Development

| Check | Command |
| --- | --- |
| Lint | `uv run --no-sync ruff check pilot` |
| Format | `uv run --no-sync ruff format --check pilot` |
| Types | `uv run --no-sync ty check` |
| Tests in bounded scopes | `bash pilot/pytest_bounded.sh` |
| Everything, as on push | `uv run --no-sync pre-commit run --all-files --hook-stage pre-push` |

`pre-commit` runs all of these on `git push`, along with a
[betterleaks](https://github.com/betterleaks/betterleaks) secret scan of the history being
pushed. The hooks check your working tree, so a dirty tree can differ from what is pushed.
The test suite takes about 7 minutes; use `SKIP=pytest git push` to skip it when you have
just run it. Building the betterleaks hook on first use requires Go.

## Documents

Validation commands and measurements are recorded next to each milestone, in date order:

1. [PILOT_RESULTS.md](PILOT_RESULTS.md): first CUDA and CPU pilot
2. [COLLECTION_PROFILE.md](COLLECTION_PROFILE.md): experience-collection profiling
3. [RESUME_VALIDATION.md](RESUME_VALIDATION.md): exact resume contract
4. [PLANNER_VALIDATION.md](PLANNER_VALIDATION.md): joint worker and planner training
5. [EVALUATION_VALIDATION.md](EVALUATION_VALIDATION.md): held-out evaluation
6. [PAPER_RECONCILIATION.md](PAPER_RECONCILIATION.md): this code versus the paper and upstream configs
7. [BOUNDED_EXPERIMENT.md](BOUNDED_EXPERIMENT.md): three-seed workflow validation
8. [PAPER_MILESTONE.md](PAPER_MILESTONE.md): paper-aligned profile
9. [PRETRAINING_MILESTONE.md](PRETRAINING_MILESTONE.md): productive worker pretraining
10. [TAX_COMPARISON.md](TAX_COMPARISON.md): matched tax comparison

[HANDOFF.md](HANDOFF.md) is the running project log, [RESEARCH_NOTES.md](RESEARCH_NOTES.md)
holds the original reproduction research, and [AGENTS.md](AGENTS.md) sets the workflow rules.

## Credits and citation

All credit for the AI Economist framework, the economic model, and the original results
belongs to the authors below. If you build on the ideas, please cite their work rather than
this repository.

**The paper this project tries to recreate:**

> Stephan Zheng, Alexander Trott, Sunil Srinivasa, Nikhil Naik, Melvin Gruesbeck,
> David C. Parkes, and Richard Socher. *The AI Economist: Improving Equality and
> Productivity with AI-Driven Tax Policies.* arXiv:2004.13332, 2020.
> <https://arxiv.org/abs/2004.13332>

**The later peer-reviewed study:**

> Stephan Zheng, Alexander Trott, Sunil Srinivasa, David C. Parkes, and Richard Socher.
> *The AI Economist: Taxation policy design via two-level deep multiagent reinforcement
> learning.* Science Advances 8(18), eabk2607, 2022.
> <https://doi.org/10.1126/sciadv.abk2607>

Its preprint is [arXiv:2108.02755](https://arxiv.org/abs/2108.02755). This project targets
the 2020 simulation study, not the later study's expanded experiments.

```bibtex
@article{zheng2020aieconomist,
  title   = {The {AI} Economist: Improving Equality and Productivity with {AI}-Driven Tax Policies},
  author  = {Zheng, Stephan and Trott, Alexander and Srinivasa, Sunil and Naik, Nikhil and
             Gruesbeck, Melvin and Parkes, David C. and Socher, Richard},
  journal = {arXiv preprint arXiv:2004.13332},
  year    = {2020},
  url     = {https://arxiv.org/abs/2004.13332}
}

@article{zheng2022aieconomist,
  title   = {The {AI} Economist: Taxation policy design via two-level deep multiagent
             reinforcement learning},
  author  = {Zheng, Stephan and Trott, Alexander and Srinivasa, Sunil and Parkes, David C. and
             Socher, Richard},
  journal = {Science Advances},
  volume  = {8},
  number  = {18},
  pages   = {eabk2607},
  year    = {2022},
  doi     = {10.1126/sciadv.abk2607}
}
```

**The simulator:** [salesforce/ai-economist](https://github.com/salesforce/ai-economist)
(Foundation 1.7.1, revision `a84d5f3`), copyright Salesforce.com, Inc., BSD-3-Clause.
It is fetched separately and is not included in this repository. The environment
configuration comes from its phase-one tutorial, and the model layout follows its recurrent
policy design. The PyTorch trainer here is a new implementation, not a conversion of the
original TensorFlow code, so it should not be treated as numerically equivalent to it.

## License

The code and documents in this repository are available under the [BSD 3-Clause
License](LICENSE), the same license as the upstream simulator. That license does not cover
the simulator, which keeps its own terms, or the papers, which are only cited here. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
