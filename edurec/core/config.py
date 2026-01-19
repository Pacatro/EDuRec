# Global state
state = {"verbose": False, "random_state": 42}

# MLFlow
EXPERIMENT_NAME: str = "TFM"

# Folders
RESULTS_FOLDER: str = "eval_results"
MODELS_FOLDER: str = "models"
RAW_DATA_FOLDER: str = "raw_data"
DATA_FOLDER: str = "data"

# Datasets
TARGET_COL: str = "rating"
ITEM_COL: str = "item_id"
USER_COL: str = "user_id"
TIME_COL: str = "created_at"

# Preprocessing
SELECTED_K: int = 10
BALANCE: bool = False

# Training
LR: float = 0.001
BATCH_SIZE: int = 128
EPOCHS: int = 100
PATIENCE: int = 5
DELTA: float = 0.001
TOP_K: int = 10
NUM_WORKERS: int = 2

# Eval
SEEDS: list[int] = [0, 1, 42]
K: int = 5
STATS_TEST: bool = False
