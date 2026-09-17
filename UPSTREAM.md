# Foundation source dependency

The existing `upstream/` directory is a source archive, not a nested Git checkout.
It is intentionally excluded from this local repository, along with generated
results and the virtual environment.

- Source: https://github.com/salesforce/ai-economist
- Revision: `a84d5f3fdcabb207d9fde7754d34906903b3e184`
- Foundation version: 1.7.1
- License: BSD-3-Clause, preserved in `upstream/LICENSE.txt`
- Local patch: in `ai_economist/foundation/scenarios/simple_wood_and_stone/layout_from_file.py`,
  replace both removed `np.int` aliases with the equivalent Python `int`.

To recreate on a new Linux x86_64 host, obtain that exact source revision as
`upstream/`, apply the two alias replacements, install Python 3.12, then run
`uv sync --locked`. The pilot imports Foundation directly from this directory.
The lockfile targets this machine's CUDA 12.6 wheel; Mac packaging is outside
this milestone. Do not install the upstream historical Ray/TensorFlow dependencies.

Exact resume fingerprints the actual Foundation Python files, scenario YAML,
and trainer/model/environment/checkpoint source. Copy the same source files with
the checkpoint and logs when moving runs. Code changes intentionally invalidate
exact resume until a migration is implemented and verified.
