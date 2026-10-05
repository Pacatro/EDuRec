from edurec.cli.ablation import app as ablation_app
from edurec.cli.dataset import app as dataset_app
from edurec.cli.eval import app as eval_app
from edurec.cli.generate import app as generate_app
from edurec.cli.optim import app as optim_app
from edurec.cli.testing import app as test_app
from edurec.cli.train import app as train_app

__all__ = [
    "ablation_app",
    "dataset_app",
    "eval_app",
    "generate_app",
    "optim_app",
    "test_app",
    "train_app",
]
