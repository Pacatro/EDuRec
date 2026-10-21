import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import pandas as pd
import torch
from safetensors.torch import load_file, save_file

from edurec.datasets.features.tabular import DataProcessor
from edurec.datasets.registry import RawData

# Bump whenever the preprocessing logic changes in a way that invalidates
# existing caches (e.g. deduplication or timestamp parsing changes), or when
# the pickled processor's module path changes (v14: package restructure).
CACHE_VERSION = 14
MANIFEST_FILENAME = "manifest.json"

CACHE_FILES = (
    "train.feather",
    "val.feather",
    "test.feather",
    "users.feather",
    "items.feather",
    "static_feats.safetensors",
    "processor.joblib",
    MANIFEST_FILENAME,
)


def source_fingerprint(folder: Path) -> dict[str, dict[str, int]]:
    """Size and modification time of every raw source file under ``folder``."""
    return {
        str(path.relative_to(folder)): {
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def read_manifest(folder: Path) -> dict[str, Any]:
    return json.loads((folder / MANIFEST_FILENAME).read_text(encoding="utf-8"))


def processed_cache_exists(
    folder: Path,
    expected_selection: Mapping[str, Any] | None = None,
    expected_params: Mapping[str, Any] | None = None,
) -> bool:
    """Whether a complete, format-compatible cache is available in ``folder``.

    Selection and any supplied preprocessing/source parameters must match;
    changing the user limit, seed or eligibility thresholds forces reprocessing.
    Fresh data overwrite this folder; raw sources are never modified.
    """
    if not all((folder / name).exists() for name in CACHE_FILES):
        return False

    try:
        manifest = read_manifest(folder)
    except (OSError, json.JSONDecodeError):
        return False

    if manifest.get("version") != CACHE_VERSION:
        return False

    if expected_selection is not None and manifest.get("selection") != dict(
        expected_selection
    ):
        return False

    if expected_params is not None and any(
        manifest.get(key) != value for key, value in expected_params.items()
    ):
        return False

    tensors = load_file(folder / "static_feats.safetensors")
    return {"u_static_feats", "u_cat_feats", "i_static_feats"}.issubset(tensors)


@dataclass
class ProcessedData:
    train: pd.DataFrame | None = None
    val: pd.DataFrame | None = None
    test: pd.DataFrame | None = None
    u_static_feats: torch.Tensor | None = None
    u_cat_feats: torch.Tensor | None = None
    i_static_feats: torch.Tensor | None = None
    user_features: pd.DataFrame | None = None
    item_features: pd.DataFrame | None = None
    data_processor: DataProcessor | None = None
    # Build-time only; not persisted with the cache.
    selected_raw: RawData | None = None
    reports: dict[str, Any] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        return all(
            value is not None
            for value in (
                self.train,
                self.val,
                self.test,
                self.u_static_feats,
                self.u_cat_feats,
                self.i_static_feats,
                self.user_features,
                self.item_features,
                self.data_processor,
            )
        )

    def splits(self) -> dict[str, pd.DataFrame]:
        if self.train is None or self.val is None or self.test is None:
            raise RuntimeError("Processed splits are not available.")

        return {
            "train": self.train,
            "val": self.val,
            "test": self.test,
        }

    def save(
        self,
        folder: Path,
        manifest: Mapping[str, Any] | None = None,
    ) -> None:
        folder.mkdir(parents=True, exist_ok=True)

        for split, df in self.splits().items():
            df.to_feather(folder / f"{split}.feather")

        if self.user_features is None or self.item_features is None:
            raise RuntimeError("User/item features are not available.")

        self.user_features.reset_index(drop=True).to_feather(folder / "users.feather")
        self.item_features.reset_index(drop=True).to_feather(folder / "items.feather")

        if (
            self.u_static_feats is None
            or self.u_cat_feats is None
            or self.i_static_feats is None
        ):
            raise RuntimeError("Static features are not available.")

        save_file(
            {
                "u_static_feats": self.u_static_feats.contiguous(),
                "u_cat_feats": self.u_cat_feats.contiguous(),
                "i_static_feats": self.i_static_feats.contiguous(),
            },
            folder / "static_feats.safetensors",
        )

        if self.data_processor is None:
            raise RuntimeError("Data processor is not available.")

        self.data_processor.save(folder / "processor.joblib")

        (folder / MANIFEST_FILENAME).write_text(
            json.dumps(dict(manifest or {}), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, folder: Path) -> Self:
        tensors = load_file(folder / "static_feats.safetensors")

        return cls(
            train=pd.read_feather(folder / "train.feather"),
            val=pd.read_feather(folder / "val.feather"),
            test=pd.read_feather(folder / "test.feather"),
            u_static_feats=tensors["u_static_feats"],
            u_cat_feats=tensors["u_cat_feats"],
            i_static_feats=tensors["i_static_feats"],
            user_features=pd.read_feather(folder / "users.feather"),
            item_features=pd.read_feather(folder / "items.feather"),
            data_processor=DataProcessor.load(folder / "processor.joblib"),
        )
