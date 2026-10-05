from pathlib import Path
from typing import Annotated

import typer

from edurec import settings
from edurec.datasets.synthetic import (
    DEFAULT_NUM_ITEMS,
    DEFAULT_NUM_USERS,
    DEFAULT_SEED,
    DEFAULT_TARGET_INTERACTIONS,
    SYNTHETIC_REQUIRED_FILES,
    generate_synthetic_raw,
)

app = typer.Typer(no_args_is_help=True)


@app.command(name="synthetic", help="Generate the raw synthetic dataset.")
def generate_synthetic(
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Folder where the raw CSV files are written.",
        ),
    ] = settings.RAW_DATA_FOLDER / "synthetic",
    num_users: Annotated[
        int,
        typer.Option("--num-users", "-u", min=1, help="Number of users to generate."),
    ] = DEFAULT_NUM_USERS,
    num_items: Annotated[
        int,
        typer.Option("--num-items", "-i", min=1, help="Number of items to generate."),
    ] = DEFAULT_NUM_ITEMS,
    target_interactions: Annotated[
        int,
        typer.Option(
            "--target-interactions",
            "-n",
            min=1,
            help="Approximate number of interactions to generate.",
        ),
    ] = DEFAULT_TARGET_INTERACTIONS,
    seed: Annotated[
        int,
        typer.Option("--seed", "-s", help="Random seed for reproducibility."),
    ] = DEFAULT_SEED,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Overwrite existing files."),
    ] = False,
) -> None:
    """Generate the synthetic raw dataset under ``data/raw/synthetic``.

    The generation is deterministic for a fixed ``--seed``. Existing files are
    kept unless ``--force`` is passed.
    """
    already_present = all(
        (output / name).exists() for name in SYNTHETIC_REQUIRED_FILES
    )
    if already_present and not force:
        print(f"[GENERATE] Dataset already present at {output}")
        print("[GENERATE] Use --force to regenerate it.")
        return

    path = generate_synthetic_raw(
        output,
        num_users=num_users,
        num_items=num_items,
        target_interactions=target_interactions,
        seed=seed,
        force=force,
    )

    print(
        f"[GENERATE] Synthetic dataset written to {path} "
        f"(users={num_users:,}, items={num_items:,}, "
        f"target_interactions={target_interactions:,}, seed={seed})"
    )
    for name in SYNTHETIC_REQUIRED_FILES:
        file_path = path / name
        size_mb = file_path.stat().st_size / 1e6
        print(f"[GENERATE]   {name}: {size_mb:.1f} MB")
