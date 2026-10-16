from edurec.evaluation.baselines import (
    BaselineRunner,
    baseline_labels,
    discover_baselines,
    get_baseline,
    native_baseline_runners,
    register_baseline,
)
from edurec.evaluation.proposed import eval_model
from edurec.evaluation.sota import eval_sota_models

# Import every baseline module so ``register_baseline`` runs before the CLI
# reads the registry. Adding a module under ``edurec/evaluation/baselines`` is
# enough to make a new native baseline available.
discover_baselines()

#: Snapshot of the native baseline registry (label -> runner).
NATIVE_BASELINE_RUNNERS = native_baseline_runners()

__all__ = [
    "NATIVE_BASELINE_RUNNERS",
    "BaselineRunner",
    "baseline_labels",
    "discover_baselines",
    "eval_model",
    "eval_sota_models",
    "get_baseline",
    "native_baseline_runners",
    "register_baseline",
]
