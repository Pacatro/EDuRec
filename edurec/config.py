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
MAX_SEQUENCE_LENGTH: int = 50

# Preprocessing
PROCESSED_FOLDER: str = f"{DATA_FOLDER}/processed"

# GCL
DROP_EDGES_P: float = 0.2
TAU: float = 0.1
MAX_SAMPLES_U: int = 2048
MAX_SAMPLES_I: int = 2048
LOSS_REDUCTION: str = "mean"

# Ranker
NUM_HEADS: int = 4
NUM_BLOCKS: int = 2
FF_DIM: int = 256
DROPOUT: float = 0.1
MAX_HISTORY_LEN: int = 50


# Training
LR: float = 0.001
BATCH_SIZE: int = 128
EPOCHS: int = 100
WEIGHT_DECAY: float = 1e-6
PATIENCE: int = 3
DELTA: float = 0.001
TOP_K: int = 10
NUM_WORKERS: int = 2
VAL_RATIO: float = 0.1
TEST_RATIO: float = 0.4
MIN_INTERACTIONS: int = 3
N_NEG_TRAIN: int = 4
N_NEG_VAL: int = 100
N_NEG_TEST: int = 100
SAVE_DATA: bool = False
MONITOR: str = f"val/Ndcg@{TOP_K}"
ADAPTIVE_K: bool = False
ALPHA: float = 0.05

# Eval
SEEDS: list[int] = [0, 1, 42]
STATS_TEST: bool = False
K: int = 5
