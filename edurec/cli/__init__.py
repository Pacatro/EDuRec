from .train import app as train_app
from .eval import app as eval_app
from .train_comp import app as train_comp_app
# from .predict import app as predict_app

__all__ = [
    "train_app",
    "eval_app",
    "train_comp_app",
    # "predict_app",
]
