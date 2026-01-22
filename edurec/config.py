# Global state
state = {"verbose": False, "random_state": 42, "device": "auto"}

# MLFlow
EXPERIMENT_NAME = "TFM"

# Folders
RESULTS_FOLDER = "eval_results"
MODELS_FOLDER = "models"
RAW_DATA_FOLDER = "raw_data"
DATA_FOLDER = "data"

# Datasets
TARGET_COL = "rating"
ITEM_COL = "item_id"
USER_COL = "user_id"
TIME_COL = "created_at"

# Preprocessing
SELECTED_K = 10
BALANCEl = False

# Model
EMB_DIM = 128
DROPOUT = 0.5
HIDDEN_DIMS = [256, 128, 64, 32, 16]

# Training
LR = 0.001
BATCH_SIZ = 128
EPOCH = 100
PATIENC = 5
DELTA = 0.001
TOP_ = 10
NUM_WORKER = 2
VAL_SIZE = 0.1
TEST_SIZE = 0.4
SAVE_PREPROCESSED_DATA = False

# Eval
K = 5
SEEDS = [0, 1, 42]
STATS_TEST = False
