# Global state
state = {"verbose": False, "random_state": 42, "device": "auto"}

# MLFlow
EXPERIMENT_NAME: str = "TFM"

# Folders
RESULTS_FOLDER: str = "evaluations"
MODELS_FOLDER: str = "models"
DATA_FOLDER: str = "data"

# Datasets
ITEM_COL: str = "item_id"
USER_COL: str = "user_id"
TIME_COL: str = "timestamp"
RATING_COL: str = "rating"
RELEVANT_COL: str = "relevant"
MIN_INTERACTIONS: int = 3

# Preprocessing
PROCESSED_FOLDER: str = f"{DATA_FOLDER}/processed"
REMOVE_SPARSE_USERS: bool = True
PREPROCESS_FEATURE_TYPES: tuple[str, ...] = (
    "numeric",
    "categorical",
    "text",
    "list",
    "time",
)
PREPROCESS_CACHE_VERSION: int = 3
TEXT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
TEXT_EMBEDDING_DIM: int = 384
TEXT_EMBEDDING_BATCH_SIZE: int = 32
TEXT_MAX_TOKENS: int = 256
TEXT_PREPROCESS_STRATEGY: str = "sentence-transformer"

# GCL
DROP_EDGES_P: float = 0.2
TAU: float = 0.2
LOSS_REDUCTION: str = "mean"
GNN_LAYERS: int = 2

# Ranker
NUM_HEADS: int = 4
NUM_BLOCKS: int = 2
FF_DIM: int = 512
DROPOUT: float = 0.1
MAX_HISTORY_LEN: int = 50
NUM_SCORES: int = 1

# Ghost model
EMB_DIM: int = 128

# Training
LR: float = 0.001
BATCH_SIZE: int = 256
EPOCHS: int = 100
WEIGHT_DECAY: float = 1e-5
PATIENCE: int = 10
DELTA: float = 0.001
TOP_K: int = 20
NUM_WORKERS: int = 4
VAL_RATIO: float = 0.1
TEST_RATIO: float = 0.2
N_NEG_TRAIN: int = 4
N_NEG_VAL: int = 99
N_NEG_TEST: int = 99
SAVE_DATA: bool = False
MONITOR: str = "val/Ndcg"
ADAPTIVE_K: bool = False
ALPHA: float = 0.05
COMPILE_MODEL: bool = False

# Eval
SEEDS: list[int] = [0, 1, 42]
STATS_TEST: bool = False
K: int = 5
