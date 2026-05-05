from .dataset import app as dataset_app
from .eval import app as eval_app
from .testing import app as test_app
from .train import app as train_app

__all__ = [
    "dataset_app",
    "eval_app",
    "test_app",
    "train_app",
]
