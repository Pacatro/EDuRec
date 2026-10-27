from edurec.evaluation.baselines import (
    baseline_labels,
    baseline_train_defaults,
    get_baseline,
)
from edurec.evaluation.baselines import (
    discover_baselines as _discover_baselines,
)
from edurec.evaluation.baselines.recbole import eval_recbole_baselines
from edurec.evaluation.proposed import eval_model

# Import every baseline module so ``register_baseline`` runs before the CLI
# reads the registry. Adding a module under ``edurec/evaluation/baselines`` is
# enough to make a new native baseline available.
_discover_baselines()

__all__ = [
    "baseline_labels",
    "baseline_train_defaults",
    "eval_model",
    "eval_recbole_baselines",
    "get_baseline",
]
