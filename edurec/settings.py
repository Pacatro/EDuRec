import os
from pathlib import Path
from typing import Literal

import lightning as L

# Global state
state = {"verbose": False, "random_state": None, "device": "auto"}


def seed_everything(seed: int | None) -> int | None:
    """Seed the project RNGs from a single source of truth."""
    if seed is None:
        state["random_state"] = None
        return None

    seed = int(seed)
    state["random_state"] = seed
    L.seed_everything(seed, workers=True, verbose=False)

    return seed


# Logging
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
)
EXPERIMENT_NAME: str = "TFM"

# Filenames and Folders
DATA_FOLDER: str = "data"
RESULTS_FOLDER: str = "results"
MODELS_FOLDER: str = "models"
MODEL_FILENAME = "model.pt"
MODEL_METADATA_FILENAME = "metadata.json"
METRICS_FILENAME = "metrics"
CONFIGS_FOLDER: str = "configs"
MODEL_CONFIGS_FOLDER: str = f"{CONFIGS_FOLDER}/model"
TRAIN_CONFIGS_FOLDER: str = f"{CONFIGS_FOLDER}/train"

# Datasets
ITEM_COL: str = "item_id"
USER_COL: str = "user_id"
TIME_COL: str = "timestamp"
RATING_COL: str = "rating"
MIN_INTERACTIONS: int = 3
TRAIN_NEGATIVES_PER_POSITIVE: int = 4
RAW_DATA_FOLDER = Path(DATA_FOLDER) / "raw"

MARS_REQUIRED_FILES = (
    "items_en.csv",
    "items_fr.csv",
    "users_en.csv",
    "users_fr.csv",
    "explicit_ratings_en.csv",
    "explicit_ratings_fr.csv",
    "implicit_ratings_en.csv",
    "implicit_ratings_fr.csv",
)
MARS_ZIP_URL = (
    "https://dataverse.harvard.edu/api/access/dataset/:persistentId/"
    "?persistentId=doi:10.7910/DVN/BMY3UD"
)
DORIS_REQUIRED_FILES = (
    "CourseInformationTable.xlsx",
    "CourseSelectionTable.xlsx",
    "StudentInformationTable.xlsx",
)
DORIS_ZIP_URL = "https://ndownloader.figstatic.com/files/41041415"
MOOCCUBEX_BASE_URL = "https://lfs.aminer.cn/misc/moocdata/data/mooccube2"
MOOCCUBEX_REQUIRED_FILES = ("entities/user.json", "entities/course.json")
MOOCCUBEX_MAX_INTERACTIONS: int = 500_000
ITM_REQUIRED_FILES = ("ratings.csv", "items.csv", "users.csv")
KAGGLE_ITM_DATASET = "irecsys/itmrec"
COCO_REQUIRED_FILES = (
    "course_latest.csv",
    "curriculum_lesson_chapter_latest.csv",
    "evaluate_latest.csv",
    "instructor_latest.csv",
    "teach_latest.csv",
)
COCO_MAX_INTERACTIONS: int = 500_000

# Preprocessing
PROCESSED_FOLDER: str = f"{DATA_FOLDER}/processed"
ATOMICFILES_FOLDER: str = f"{DATA_FOLDER}/atomicfiles"
REMOVE_SPARSE: bool = True
# Collapse repeated (user, item) rows before splitting. Off by default so real,
# temporally distinct events (view -> progress -> complete) are preserved; the
# datamodule still deduplicates datasets without timestamps to avoid leakage.
DEDUPLICATE_INTERACTIONS: bool = False
PREPROCESS_FEATURE_TYPES: tuple[str, ...] = (
    "numeric",
    "categorical",
    "text",
    "list",
    "time",
)
TEXT_EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"
TEXT_EMBEDDING_DIM: int = 384
TEXT_EMBEDDING_BATCH_SIZE: int = 32
TEXT_MAX_TOKENS: int = 256

# GNN
GNN_LAYERS: int = 2
GNN_HEADS: int = 4

# RecSys
SEQ_CELL: Literal["gru", "lstm"] = "gru"
GRU_HIDDEN_DIM: int = 128
GRU_LAYERS: int = 1
TRANSFORMER_HIDDEN_DIM: int = 128
TRANSFORMER_LAYERS: int = 2
TRANSFORMER_HEADS: int = 4
DROPOUT: float = 0.15
MAX_HISTORY_LEN: int = 50
# The main architecture reads the last valid hidden state; attention pooling is
# kept only as an optional ablation.
USE_ATTENTION_POOLING: bool = False
# Feature toggles for the kg_rnn architecture.
USE_ITEM_FEATURES: bool = True
USE_INTERACTION_FEATURES: bool = True
USE_TIME_FEATURES: bool = True

# Embeddings
EMB_DIM: int = 128

# Training
LR: float = 2e-4
WEIGHT_DECAY: float = 1e-4
BATCH_SIZE: int = 256
PATIENCE: int = 5
TOP_K: int = 20
EPOCHS: int = 150
DELTA: float = 0.001
NUM_WORKERS: int = 4
VAL_RATIO: float = 0.1
TEST_RATIO: float = 0.2
SAVE_DATA: bool = False
ADAPTIVE_K: bool = False
COMPILE_MODEL: bool = False
TOP_KS: list[int] = [5, 10, 20]

# SOTA MODELS
SOTA_MODELS: list[str] = [
    # Baselines
    "ItemKNN",
    # Collaborative filtering / neural CF
    "NeuMF",
    "LightGCN",
    # Autoencoder / linear models
    "MultiVAE",
    # Advanced graph / contrastive models
    "SGL",
    # Sequential models
    "SASRec",
    "BERT4Rec",
]
SOTA_GPU_ID: int = 1

# Hyperparameter optimization
OPTIM_N_TRIALS: int = 30
