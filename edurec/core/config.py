# Global state
state = {"verbose": False}

# Folders
RESULTS_FOLDER: str = "results"
MODELS_FOLDER: str = "saved_models"

# Datasets
BALANCE: bool = False
TARGET_COL: str = "rating"
ITEM_COL: str = "item_id"
USER_COL: str = "user_id"

# Training
LR: float = 0.001
BATCH_SIZE: int = 128
EPOCHS: int = 50
PATIENCE: int = 5
DELTA: float = 0.001
TOP_K: int = 10

# Eval
SEEDS: list[int] = [0, 1, 42]
K: int = 5
STATS_TEST: bool = False
