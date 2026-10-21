# Repository Guidelines

## Project Structure & Module Organization

EDuRec is a Python 3.12+ research project for e-learning recommendations, built with PyTorch, PyTorch Geometric, and Lightning. Source lives in `edurec/`: `cli/` defines Typer commands; `datasets/` handles loading, preprocessing, caching, and user histories; `recsys/` contains training, ranking, optimization, and model architectures; `evaluation/` implements benchmarks and ablations. Shared defaults live in `edurec/settings.py`.

Keep dataset-specific configuration in `configs/<arch>-<dataset>.yaml`. Tests live in `tests/`, exploratory notebooks in `notebooks/`, and the architecture illustration in `model-diagram.png`. Local datasets, checkpoints, and experiment outputs belong in ignored directories such as `data/`, `models/`, and `results/`.

## Build, Test, and Development Commands

Run commands from the repository root:

- `uv sync --group dev`: install runtime and development dependencies from the lockfile.
- `uv build`: build the Python distribution with Hatchling.
- `uv run edurec --help`: inspect available commands and options.
- `uv run python -m pytest -q`: run the CPU-based test suite.
- `uv run ruff check .`: check Python lint issues.
- `uv run pyright edurec`: check source types.
- `uv run edurec --device cpu --random-state 42 train --dataset doris --arch kg_rnn --debug`: run a reproducible training smoke check; dataset preparation may download data and pretrained weights.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for functions and modules, `PascalCase` for classes, and uppercase constants. Follow adjacent code's formatting and imports; `pyproject.toml` has no custom Ruff configuration. Annotate public interfaces and document tensor shapes where useful. Keep tensors device-agnostic and expose experiment parameters through configuration rather than hardcoding them.

## Testing Guidelines

Use pytest with `test_*.py` files and `test_*` functions. Shared fixtures belong in `tests/conftest.py`. Use small synthetic CPU datasets without downloading datasets or model weights. Cover affected contracts, especially temporal splits, leakage prevention, tensor shapes, ranking, and CLI behavior. Run focused checks with `uv run python -m pytest tests/test_datasets.py -q`. No coverage threshold is configured.

## Commit & Pull Request Guidelines

Follow the history's scoped Conventional Commits, such as `feat(datasets): ...`, `test(datasets): ...`, and `chore(notebooks): ...`. Keep commits focused. PRs should explain the problem, resulting behavior, validation commands, and configuration or cache changes. Link related issues and include plots when changing visualizations.

## Data & Configuration Integrity

Preserve temporal evaluation boundaries and fit preprocessing on training data. Never commit datasets, model weights, credentials, or experiment databases.
