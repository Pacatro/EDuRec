import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import joblib
import pandas as pd
import torch
from safetensors.torch import load_file, save_file
from sklearn.compose import ColumnTransformer

from edurec import settings
from edurec.datasets.config import BuildManifest
from edurec.datasets.sources.registry import Schema

REPORT_FILENAME = "report.json"
MANIFEST_FILENAME = "manifest.json"


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
    """Canonical, fully-built data consumed by models and exported views."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    u_static_feats: torch.Tensor
    u_cat_feats: torch.Tensor
    i_static_feats: torch.Tensor
    user_features: pd.DataFrame
    item_features: pd.DataFrame
    schema: Schema
    id_maps: dict[str, dict[object, int]]
    transforms: dict[str, ColumnTransformer | None]
    graph_node_features: dict[str, pd.DataFrame] = field(default_factory=dict)
    reports: dict[str, Any] = field(default_factory=dict)
    manifest: BuildManifest | None = None

    @property
    def dataset_name(self) -> str:
        return "" if self.manifest is None else self.manifest.dataset

    @property
    def num_users(self) -> int:
        return self.u_static_feats.shape[0]

    @property
    def num_items(self) -> int:
        return self.i_static_feats.shape[0]

    @property
    def num_interactions(self) -> int:
        return sum(len(frame) for frame in self.splits().values())

    @property
    def sparsity(self) -> float:
        if self.num_users == 0 or self.num_items == 0:
            return 0.0

        return 1 - (self.num_interactions / (self.num_users * self.num_items))

    @property
    def is_explicit(self) -> bool:
        return settings.RATING_COL in self.train.columns

    @property
    def has_temporal_order(self) -> bool:
        return settings.TIME_COL in self.train.columns

    @property
    def feedback_type(self) -> str:
        return "explicit" if self.is_explicit else "implicit"

    @property
    def split_strategy(self) -> str:
        return "temporal" if self.has_temporal_order else "random"

    def splits(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def positive_splits(self) -> dict[str, pd.DataFrame]:
        """Validated positive interactions per split."""
        name = self.dataset_name or "processed data"
        relevant: dict[str, pd.DataFrame] = {}

        for split, frame in self.splits().items():
            positive = frame.loc[frame[settings.RELEVANT_COL] > 0].reset_index(
                drop=True
            )
            if positive.empty:
                raise ValueError(
                    f"{name}: {split} split has no positive interactions. "
                    "Check filtering, relevance thresholds and split ratios."
                )
            for column, size in (
                (settings.USER_COL, self.num_users),
                (settings.ITEM_COL, self.num_items),
            ):
                if not positive[column].between(0, size - 1).all():
                    raise ValueError(
                        f"{name}: {split} contains invalid {column} values."
                    )
            relevant[split] = positive

        return relevant

    def save(self, folder: Path) -> None:
        replace_folder(folder, self._write)

    def _write(self, folder: Path) -> None:
        folder.mkdir(parents=True)

        for split, df in self.splits().items():
            df.to_feather(folder / f"{split}.feather")

        self.user_features.reset_index(drop=True).to_feather(folder / "users.feather")
        self.item_features.reset_index(drop=True).to_feather(folder / "items.feather")

        save_file(
            {
                "u_static_feats": self.u_static_feats.contiguous(),
                "u_cat_feats": self.u_cat_feats.contiguous(),
                "i_static_feats": self.i_static_feats.contiguous(),
            },
            folder / "static_feats.safetensors",
        )

        joblib.dump(
            {
                "schema": self.schema,
                "id_maps": self.id_maps,
                "transforms": self.transforms,
                "graph_node_features": self.graph_node_features,
            },
            folder / "preprocessing.joblib",
        )

        (folder / REPORT_FILENAME).write_text(
            json.dumps(self.reports, indent=2, sort_keys=True, default=_to_builtin),
            encoding="utf-8",
        )
        if self.manifest is not None:
            (folder / MANIFEST_FILENAME).write_text(
                json.dumps(self.manifest.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )

    @classmethod
    def load(cls, folder: Path) -> Self:
        if not (folder / "preprocessing.joblib").exists():
            raise ValueError(
                "Legacy preprocessing cache: rebuild without --use-processed."
            )
        preprocessing = joblib.load(folder / "preprocessing.joblib")
        tensors = load_file(folder / "static_feats.safetensors")
        report_path = folder / REPORT_FILENAME
        # Caches written before reports existed contain no report file.
        reports = (
            json.loads(report_path.read_text(encoding="utf-8"))
            if report_path.exists()
            else {}
        )
        manifest_path = folder / MANIFEST_FILENAME
        manifest = (
            BuildManifest.from_dict(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
            if manifest_path.exists()
            else None
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
            schema=preprocessing["schema"],
            id_maps=preprocessing["id_maps"],
            transforms=preprocessing["transforms"],
            graph_node_features=preprocessing["graph_node_features"],
            reports=reports,
            manifest=manifest,
        )
