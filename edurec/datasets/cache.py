from dataclasses import dataclass
from pathlib import Path
from typing import Self

import pandas as pd
import torch
from safetensors.torch import load_file, save_file

from .dataprocessor import DataProcessor

CACHE_FILES = (
    "train.feather",
    "val.feather",
    "test.feather",
    "static_feats.safetensors",
    "processor.joblib",
)


def processed_cache_exists(folder: Path) -> bool:
    if not all((folder / name).exists() for name in CACHE_FILES):
        return False

    tensors = load_file(folder / "static_feats.safetensors")
    return {
        "u_static_feats",
        "i_static_feats",
        "user_stats",
        "item_stats",
    }.issubset(tensors)


@dataclass
class ProcessedData:
    train: pd.DataFrame | None = None
    val: pd.DataFrame | None = None
    test: pd.DataFrame | None = None
    u_static_feats: torch.Tensor | None = None
    i_static_feats: torch.Tensor | None = None
    user_stats: torch.Tensor | None = None
    item_stats: torch.Tensor | None = None
    data_processor: DataProcessor | None = None

    @property
    def is_ready(self) -> bool:
        return all(
            value is not None
            for value in (
                self.train,
                self.val,
                self.test,
                self.u_static_feats,
                self.i_static_feats,
                self.user_stats,
                self.item_stats,
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
        folder.mkdir(parents=True, exist_ok=True)

        for split, df in self.splits().items():
            df.to_feather(folder / f"{split}.feather")

        if (
            self.u_static_feats is None
            or self.i_static_feats is None
            or self.user_stats is None
            or self.item_stats is None
        ):
            raise RuntimeError("Static features or router stats are not available.")

        save_file(
            {
                "u_static_feats": self.u_static_feats.contiguous(),
                "i_static_feats": self.i_static_feats.contiguous(),
                "user_stats": self.user_stats.contiguous(),
                "item_stats": self.item_stats.contiguous(),
            },
            folder / "static_feats.safetensors",
        )

        if self.data_processor is None:
            raise RuntimeError("Data processor is not available.")

        self.data_processor.save(folder / "processor.joblib")

    @classmethod
    def load(cls, folder: Path) -> Self:
        tensors = load_file(folder / "static_feats.safetensors")

        return cls(
            train=pd.read_feather(folder / "train.feather"),
            val=pd.read_feather(folder / "val.feather"),
            test=pd.read_feather(folder / "test.feather"),
            u_static_feats=tensors["u_static_feats"],
            i_static_feats=tensors["i_static_feats"],
            user_stats=tensors["user_stats"],
            item_stats=tensors["item_stats"],
            data_processor=DataProcessor.load(folder / "processor.joblib"),
        )
