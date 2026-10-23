import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import pandas as pd
import torch
from safetensors.torch import load_file, save_file

from edurec.datasets.features.tabular import DataProcessor
from edurec.datasets.registry import RawData

REPORT_FILENAME = "report.json"


def replace_folder[T](folder: Path, write: Callable[[Path], T]) -> T:
    """Build ``folder`` in a sibling staging directory, then swap it in.

    Readers see either the previous folder or the complete new one. If ``write``
    fails, the previous folder is left untouched.
    """
    staging = folder.with_name(f"{folder.name}.tmp")
    if staging.exists():
        shutil.rmtree(staging)
    try:
        result = write(staging)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if folder.exists():
        shutil.rmtree(folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(folder)
    return result


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


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

    def save(self, folder: Path) -> None:
        replace_folder(folder, self._write)

    def _write(self, folder: Path) -> None:
        folder.mkdir(parents=True)

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

        (folder / REPORT_FILENAME).write_text(
            json.dumps(self.reports, indent=2, sort_keys=True, default=_to_builtin),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, folder: Path) -> Self:
        tensors = load_file(folder / "static_feats.safetensors")
        report_path = folder / REPORT_FILENAME
        # Caches written before reports existed contain no report file.
        reports = (
            json.loads(report_path.read_text(encoding="utf-8"))
            if report_path.exists()
            else {}
        )

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
            reports=reports,
        )
