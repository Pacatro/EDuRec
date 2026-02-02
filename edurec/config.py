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

# Preprocessing
SELECTED_K: int = 10
BALANCE: bool = False

# Model
EMB_DIM: int = 256
DROPOUT: float = 0.2
HIDDEN_DIMS: list[int] = [256, 256, 128]

# Training
LR: float = 0.001
BATCH_SIZE: int = 128
EPOCHS: int = 100
PATIENCE: int = 3
DELTA: float = 0.001
TOP_K: int = 10
NUM_WORKERS: int = 2
VAL_SIZE: float = 0.1
TEST_SIZE: float = 0.4
N_NEG_TRAIN: int = 4
N_NEG_VAL: int = 100
N_NEG_TEST: int = 100
SAVE_DATA: bool = False
MONITOR: str = "val/Loss"
# If alpha is 0.5, both tasks have the same importance
# If alpha is 0.0, only relevance task is considered
# If alpha is 1.0, only rating task is considered
ALPHA: float = 0.5

# Eval
SEEDS: list[int] = [0, 1, 42]
STATS_TEST: bool = False
K: int = 5
