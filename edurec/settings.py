from pathlib import Path

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


# W&B
EXPERIMENT_NAME: str = "TFM"

# Filenames and Folders
DATA_FOLDER: str = "data"
RESULTS_FOLDER: str = "results"
MODELS_FOLDER: str = "models"
MODEL_FILENAME = "model.pt"
RECBOLE_INTER_FILES_FOLDER: str = f"{DATA_FOLDER}/inter"
MODEL_METADATA_FILENAME = "metadata.json"
METRICS_FILENAME = "metrics.csv"
CONFIGS_FOLDER: str = "configs"

# Datasets
ITEM_COL: str = "item_id"
USER_COL: str = "user_id"
TIME_COL: str = "timestamp"
RATING_COL: str = "rating"
RELEVANT_COL: str = "relevant"
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

MOOCCUBEX_REQUIRED_FILES = (
    "entities/user.json",
    "entities/course.json",
)

ITM_REQUIRED_FILES = ("ratings.csv", "items.csv", "users.csv")

KAGGLE_ITM_DATASET = "irecsys/itmrec"

# Preprocessing
PROCESSED_FOLDER: str = f"{DATA_FOLDER}/processed"
ATOMICFILES_FOLDER: str = f"{DATA_FOLDER}/atomicfiles"
REMOVE_SPARSE: bool = True
PREPROCESS_FEATURE_TYPES: tuple[str, ...] = (
    "numeric",
    "categorical",
    "text",
    "list",
    "time",
)
PREPROCESS_CACHE_VERSION: int = 4
TEXT_EMBEDDING_MODEL: str = "paraphrase-MiniLM-L3-v2"
TEXT_EMBEDDING_DIM: int = 384
TEXT_EMBEDDING_BATCH_SIZE: int = 32
TEXT_MAX_TOKENS: int = 256
TEXT_PREPROCESS_STRATEGY: str = "sentence-transformer"

# GCL
DROP_EDGES_P: float = 0.2
TAU: float = 0.15
LOSS_REDUCTION: str = "mean"
GNN_LAYERS: int = 2

# RecSys
NUM_HEADS: int = 4
NUM_BLOCKS: int = 2
FF_DIM: int = 512
DROPOUT: float = 0.15
MAX_HISTORY_LEN: int = 50

# Embeddings
EMB_DIM: int = 128

# Training
LR: float = 2e-4
WEIGHT_DECAY: float = 1e-4
BATCH_SIZE: int = 128
PATIENCE: int = 5
TOP_K: int = 20
MONITOR_METRIC: str = f"val/ndcg@{TOP_K}"
EPOCHS: int = 150
DELTA: float = 0.001
NUM_WORKERS: int = 4
VAL_RATIO: float = 0.1
TEST_RATIO: float = 0.2
SAVE_DATA: bool = False
ADAPTIVE_K: bool = False
LOSS_ALPHA: float = 0.05
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

# Hyperparameter optimization
OPTIM_N_TRIALS: int = 30
