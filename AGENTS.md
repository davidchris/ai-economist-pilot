# Project workflow

- Work in this local repository. Preserve historical results and the pinned `upstream/` checkout; do not synchronize to another machine unless requested.
- Use `uv` for Python dependency management and execution. Reuse `.venv`; dependencies are declared in `pyproject.toml` and resolved in `uv.lock`.
- Use Ruff for linting and formatting: `uv run --no-sync ruff check pilot` and `uv run --no-sync ruff format --check pilot`.
- Use ty for type checking: `uv run --no-sync ty check`.
- Use pytest for testing: `uv run --no-sync pytest`. Run relevant checks after changes. Training and environment tests must run within a systemd scope limited to CPUQuota=200%, MemoryMax=8G, MemorySwapMax=0, with a 600-second timeout and single-thread BLAS/OpenMP settings.
- Recheck disk, RAM, load, and GPU availability before experiments. Keep at least 15 GiB free. Do not change other services or other disks on the host.
- Preserve Foundation economic mechanics, numerical checks, and exact update-boundary restart semantics. Document any compatibility limits. Evaluation and planner training are separate future milestones.
- Keep generated results, virtual environments, caches, and the pinned upstream source archive out of this repository. Its provenance and compatibility patch are documented in `UPSTREAM.md`. Record validation commands and measured results in tracked documentation.
- Keep tracked files free of hostnames, home paths, personal emails, and machine-specific operational details: this repository is public. `pre-commit` runs Ruff, ty, betterleaks, and pytest on `git push`. Never push with `--mirror` or `--all`: local `refs/t3/*` and `refs/backup/*` keep unscrubbed history.
