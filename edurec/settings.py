# Global state
state = {"verbose": False, "random_state": 42, "device": "auto"}

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
