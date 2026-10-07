# Repository Guidelines

## Project Structure & Module Organization

EDuRec uses Python 3.12+, PyTorch, PyTorch Geometric, and Lightning. Source lives in `edurec/`: `cli/` contains Typer commands, `datasets/` handles loading, preprocessing, caching, and histories, `recsys/archs/` defines models, and `evaluation/` implements benchmarks and ablations. Shared defaults live in `edurec/settings.py`.

Keep dataset-specific YAML files in `configs/model/` and `configs/train/`, named `config-<dataset>-<arch>.yaml` and `train-config-<dataset>-<arch>.yaml` respectively (matching the copies in `results/optimization/`). `notebooks/` contains experiment analysis; `model-diagram.png` illustrates the architecture. Local datasets, checkpoints, and results belong in the ignored `data/`, `models/`, and `results/` directories.

## Build, Test, and Development Commands

- `uv sync --group dev`: install runtime and development dependencies.
- `uv build`: build the Python distribution using Hatchling.
- `uv run edurec --help`: inspect CLI commands and options.
- `uv run edurec --device cpu --random-state 42 train --dataset doris --arch sasrec_text --debug`: run a reproducible training smoke check.
- `uv run pyright --pythonpath .venv/bin/python --pythonversion 3.12 edurec`: run static type checks.
- `uv run python -m unittest discover -s tests -v`: run contributor-added unittest tests.

Initial dataset preparation may download datasets and pretrained text weights.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for functions/modules, `PascalCase` for classes, and uppercase constants. Add type annotations to public interfaces and document tensor shapes where they clarify model behavior. Follow adjacent code's import ordering and formatting; no formatter is pinned in `pyproject.toml`. Keep tensors device-agnostic and configuration in dataclasses or YAML rather than hardcoded training parameters.

## Testing Guidelines

There are currently no tracked automated tests or enforced coverage threshold. Add focused regression tests under `tests/test_<component>.py` using standard-library `unittest`; pytest is not declared as a dependency. Prioritize chronological leakage, feature alignment, negative sampling, finite losses/gradients, and candidate/full-catalog scoring consistency. Gate CUDA tests on availability and explain skipped checks.

## Commit & Pull Request Guidelines

Recent commits use scoped conventional messages, such as `feat(recsys): ...`, `fix(configs): ...`, and `refactor(cli): ...`. Keep commits focused. PR descriptions should explain the problem, resulting behavior, commands run, and configuration or cache changes. Link related issues when available; include plots for changes to experiment visualizations.

## Data & Configuration Integrity

Fit interaction preprocessing on training data and preserve the evaluation protocol. Increment `CACHE_VERSION` when changing cached preprocessing semantics. Explicit CLI values override saved configurations, then defaults. Never commit datasets, model weights, credentials, or experiment databases.
