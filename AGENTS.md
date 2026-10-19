# Repository Guidelines

## Project Structure & Module Organization

EDuRec uses Python 3.12+, PyTorch, PyTorch Geometric, and Lightning. Source lives in `edurec/`: `cli/` contains Typer commands, `datasets/` handles loading, preprocessing, caching, and histories, `recsys/archs/` defines models, and `evaluation/` implements benchmarks and ablations. Shared defaults live in `edurec/settings.py`.

Keep dataset-specific YAML files in `configs/`, named `<arch>-<dataset>.yaml`, with top-level `arch`, `model`, and `train` fields (matching the copies in `results/optimization/`). `notebooks/` contains experiment analysis; `model-diagram.png` illustrates the architecture. Local datasets, checkpoints, and results belong in the ignored `data/`, `models/`, and `results/` directories.

## Build, Test, and Development Commands

- `uv sync --group dev`: install runtime and development dependencies.
- `uv build`: build the Python distribution using Hatchling.
- `uv run edurec --help`: inspect CLI commands and options.
- `uv run edurec --device cpu --random-state 42 train --dataset doris --arch sasrec_text --debug`: run a reproducible training smoke check.
- `uv run pyright --pythonpath .venv/bin/python --pythonversion 3.12 edurec`: run static type checks.

Initial dataset preparation may download datasets and pretrained text weights.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for functions/modules, `PascalCase` for classes, and uppercase constants. Add type annotations to public interfaces and document tensor shapes where they clarify model behavior. Follow adjacent code's import ordering and formatting; no formatter is pinned in `pyproject.toml`. Keep tensors device-agnostic and configuration in dataclasses or YAML rather than hardcoded training parameters.

## Commit & Pull Request Guidelines

Recent commits use scoped conventional messages, such as `feat(recsys): ...`, `fix(configs): ...`, and `refactor(cli): ...`. Keep commits focused. PR descriptions should explain the problem, resulting behavior, commands run, and configuration or cache changes. Link related issues when available; include plots for changes to experiment visualizations.

## Data & Configuration Integrity

Fit interaction preprocessing on training data and preserve the evaluation protocol. Increment `CACHE_VERSION` when changing cached preprocessing semantics. Explicit CLI values override saved configurations, then defaults. Never commit datasets, model weights, credentials, or experiment databases.
