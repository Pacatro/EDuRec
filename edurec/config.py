# Global state
state = {"verbose": False, "random_state": 42, "device": "auto"}

# W&B
EXPERIMENT_NAME: str = "TFM"

# Filenames and Folders
RESULTS_FOLDER: str = "evaluations"
MODELS_FOLDER: str = "models"
DATA_FOLDER: str = "data"
MODEL_FILENAME = "model.pt"
MODEL_CONFIG_FILENAME = "config.json"
METRICS_FILENAME = "metrics.csv"

# Datasets
ITEM_COL: str = "item_id"
USER_COL: str = "user_id"
TIME_COL: str = "timestamp"
INTERACTION_ORDER_COL: str = "interaction_order"
RATING_COL: str = "rating"
RELEVANT_COL: str = "relevant"
CANDIDATE_IDS_COL: str = "candidate_ids"
CANDIDATE_LABELS_COL: str = "candidate_labels"
POSITIVE_POSITION_COL: str = "positive_position"
MIN_INTERACTIONS: int = 3

# Preprocessing
PROCESSED_FOLDER: str = f"{DATA_FOLDER}/processed"
REMOVE_SPARSE: bool = True
PREPROCESS_FEATURE_TYPES: tuple[str, ...] = (
    "numeric",
    "categorical",
    "text",
    "list",
    "time",
)
PREPROCESS_CACHE_VERSION: int = 4
TEXT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
TEXT_EMBEDDING_DIM: int = 384
TEXT_EMBEDDING_BATCH_SIZE: int = 32
TEXT_MAX_TOKENS: int = 256
TEXT_PREPROCESS_STRATEGY: str = "sentence-transformer"

# GCL
DROP_EDGES_P: float = 0.1
TAU: float = 0.2
LOSS_REDUCTION: str = "mean"
GNN_LAYERS: int = 2

# Ranker
NUM_HEADS: int = 4
NUM_BLOCKS: int = 2
FF_DIM: int = 256
DROPOUT: float = 0.1
MAX_HISTORY_LEN: int = 30
NUM_SCORES: int = 1

# Embeddings
EMB_DIM: int = 128

# Training (Retreival)
RETRIEVAL_EPOCHS: int = 40
RETRIEVAL_LR: float = 0.001
RETRIEVAL_WEIGHT_DECAY: float = 1e-4
RETRIEVAL_BATCH_SIZE: int = 256
RETRIEVAL_PATIENCE: int = 5
RETRIEVAL_TOP_K: int = 50

# Training (Ranker)
RANKER_LR: float = 3e-4
RANKER_WEIGHT_DECAY: float = 2e-4
RANKER_BATCH_SIZE: int = 128
RANKER_EPOCHS: int = 30
RANKER_PATIENCE: int = 5
RANKER_TOP_K: int = 20

# Training (General)
TOP_N: int = 50
DELTA: float = 0.0005
NUM_WORKERS: int = 4
VAL_RATIO: float = 0.2
TEST_RATIO: float = 0.4
SAVE_DATA: bool = False
ADAPTIVE_K: bool = False
ALPHA: float = 0.01
COMPILE_MODEL: bool = False

# Eval
SEEDS: list[int] = [0, 1, 42]
STATS_TEST: bool = False
K: int = 5
