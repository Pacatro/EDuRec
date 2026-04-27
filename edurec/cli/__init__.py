from .dataset import app as dataset_app
from .testing import app as test_app
from .train import app as train_app

__all__ = [
    "train_app",
    "test_app",
    "dataset_app",
]
