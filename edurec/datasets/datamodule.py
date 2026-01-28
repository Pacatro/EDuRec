import multiprocessing
from itertools import chain
from pathlib import Path
from typing import Any, cast

import lightning as L
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .. import config
from .loaders import DatasetName, load_data
from .preprocessor import Preprocessor
from .utils import get_column_types, global_preprocessing, process_chunk


class ElearningDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy().map(lambda x: torch.tensor(x))
        self.columns = df.columns

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return cast(dict[str, torch.Tensor], self.df.iloc[idx].to_dict())


class ElearningDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: DatasetName,
        batch_size: int,
        test_size: float,
        val_size: float,
        negative_samples: int = config.NEG_SAMPLES,
        save_data: bool = False,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.save_data = save_data
        self.negative_samples = negative_samples

        self.id_cols = [config.USER_COL, config.ITEM_COL]
        self.numeric_cols: list[str] = []
        self.categorical_lengths: dict[str, int] = {}

        self.processed_path = (
            Path(config.DATA_FOLDER) / "preprocessed" / f"{self.dataset_name.value}.csv"
        )

        self.df = load_data(dataset)
        print(self.df.shape)
        self._process_data()

    @staticmethod
    def _item_static_columns(
        df: pd.DataFrame, item_col: str, exclude: set[str]
    ) -> list[str]:
        """
        Columnas que parecen ser "propias del ítem" (constantes para cada item_id).
        Sirve para rellenar features del ítem cuando fabricamos una interacción negativa.
        """
        cols = [c for c in df.columns if c not in exclude]
        item_static: list[str] = []
        # Heurística: nunique por item <= 1
        for c in cols:
            nun = df.groupby(item_col)[c].nunique(dropna=False)
            if int(nun.max()) <= 1:
                item_static.append(c)
        return item_static

    @staticmethod
    def _build_seen_items(
        df: pd.DataFrame, user_col: str, item_col: str
    ) -> dict[int, set[int]]:
        seen: dict[int, set[int]] = {}
        for u, g in df.groupby(user_col):
            assert isinstance(u, int)
            seen[u] = set(map(int, g[item_col].tolist()))
        return seen

    def _split(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if config.TIME_COL in self.df.columns:
            self.df = self.df.sort_values(by=config.TIME_COL)
            train_val_df, test_df = train_test_split(
                self.df,
                test_size=self.test_size,
                shuffle=False,
                random_state=self.random_state,
            )
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=self.val_size,
                shuffle=False,
                random_state=self.random_state,
            )
        else:
            train_val_df, test_df = train_test_split(
                self.df,
                test_size=self.test_size,
                random_state=self.random_state,
            )
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=self.val_size / (1 - self.test_size),
                random_state=self.random_state,
            )

        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)

        return train_df, val_df, test_df

    def _negative_sampling(
        self,
        df_pos: pd.DataFrame,
        train_seen: dict[int, set[int]],
        all_items: np.ndarray,
        item_features: dict[int, dict],
        n_neg: int,
        neg_rating: float,
        n_jobs: int = -1,
    ) -> pd.DataFrame:
        """
        Para cada fila (u, i_pos, rating, features...), añade n_neg filas negativas:
        (u, i_neg) con is_relevant=0, rating=neg_rating y features del ítem copiadas.
        También añade is_relevant para la fila positiva.
        """

        if config.RELEVANT_COL not in df_pos.columns:
            raise ValueError(
                f"'{config.RELEVANT_COL}' no está en df_pos, pero has indicado que ya viene incluido."
            )

        num_cores = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs

        chunks = np.array_split(df_pos, num_cores)

        seeds = np.random.randint(0, 1_000_000, size=num_cores)

        raw_results = Parallel(n_jobs=num_cores)(
            delayed(process_chunk)(
                chunks[i],
                train_seen,
                all_items,
                item_features,
                n_neg,
                neg_rating,
                seeds[i],
            )
            for i in range(num_cores)
        )

        if raw_results is None:
            raise ValueError("No results were generated")

        results = cast(list[list[dict[str, Any]]], raw_results)

        flat_results = list(chain.from_iterable(results))

        return pd.DataFrame(flat_results)

    def _generate_neg_samples(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        neg_rating = float(train_df[config.RATING_COL].min())
        all_items = np.array(train_df[config.ITEM_COL].unique(), dtype=np.int64)
        train_seen = self._build_seen_items(train_df, config.USER_COL, config.ITEM_COL)
        exclude = {config.USER_COL, config.ITEM_COL, config.RATING_COL}
        item_static_cols = self._item_static_columns(train_df, config.ITEM_COL, exclude)

        if item_static_cols:
            item_feat_df = train_df[
                [config.ITEM_COL] + item_static_cols
            ].drop_duplicates(config.ITEM_COL)
            item_features = {
                int(row[config.ITEM_COL]): {c: row[c] for c in item_static_cols}
                for _, row in item_feat_df.iterrows()
            }
        else:
            item_features = {}

        val_df = self._negative_sampling(
            df_pos=val_df,
            train_seen=train_seen,
            all_items=all_items,
            item_features=item_features,
            n_neg=self.negative_samples,
            neg_rating=neg_rating,
        )

        test_df = self._negative_sampling(
            df_pos=test_df,
            train_seen=train_seen,
            all_items=all_items,
            item_features=item_features,
            n_neg=self.negative_samples,
            neg_rating=neg_rating,
        )

        return val_df, test_df

    def _process_data(self) -> None:
        threshold = self.threshold
        global_preprocessing(self.df, threshold)

        train_df, val_df, test_df = self._split()

        val_df, test_df = self._generate_neg_samples(train_df, test_df, val_df)

        self.numeric_cols, self.categorical_lengths = get_column_types(
            train_df, self.id_cols
        )

        self.preprocessor = Preprocessor(
            self.numeric_cols,
            list(self.categorical_lengths.keys()),
            self.id_cols,
        )

        self.train_df, self.val_df, self.test_df = self.preprocessor.fit_transform(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

        if self.test_df is None:
            return

        # TODO: Find a way to not concatenate the data
        self.df = pd.concat(
            [self.train_df, self.val_df, self.test_df], ignore_index=True
        )

        if self.save_data:
            # Save the preprocessed data
            self.processed_path.parent.mkdir(parents=True, exist_ok=True)
            self.df.to_csv(self.processed_path, index=False)

    def setup(self, stage: str | None = None) -> None:
        match stage:
            case "fit":
                self.train_ds = ElearningDataset(self.train_df)
                self.val_ds = ElearningDataset(self.val_df)
            case "test":
                self.test_ds = (
                    ElearningDataset(self.test_df) if self.test_df is not None else None
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, num_workers=config.NUM_WORKERS
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, num_workers=config.NUM_WORKERS
        )

    def test_dataloader(self) -> DataLoader | None:
        if self.test_ds is None:
            return None

        return DataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=config.NUM_WORKERS
        )

    @property
    def num_users(self) -> int:
        return int(self.df[config.USER_COL].nunique())

    @property
    def num_items(self) -> int:
        return int(self.df[config.ITEM_COL].nunique())

    @property
    def numeric_features(self) -> list[str]:
        return self.numeric_cols

    @property
    def cat_cardinalities(self) -> dict[str, int]:
        return {k: v + 2 for k, v in self.categorical_lengths.items()}

    @property
    def sparsity(self) -> float:
        return 1 - len(self.df) / (self.num_users * self.num_items)

    @property
    def threshold(self) -> float:
        return float(self.df[config.RATING_COL].mean())

    @property
    def min_rating(self) -> float:
        return float(self.df[config.RATING_COL].min())

    @property
    def max_rating(self) -> float:
        return float(self.df[config.RATING_COL].max())
