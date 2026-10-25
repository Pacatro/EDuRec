"""Select adapters without adding model branches to the data module."""

from collections.abc import Callable
from typing import Any

from edurec.datasets.adapters.base import ModelDataAdapter
from edurec.datasets.adapters.proposed import ProposedAdapter
from edurec.datasets.cache import ProcessedData
from edurec.datasets.queries import QueryConfig

AdapterFactory = Callable[[ProcessedData, QueryConfig], ModelDataAdapter[Any, Any]]
_FACTORIES: dict[str, AdapterFactory] = {"proposed": ProposedAdapter}


def register_adapter(name: str, factory: AdapterFactory) -> None:
    """Register a class or factory; reject accidental replacement."""
    if not name or name in _FACTORIES:
        raise ValueError(f"Adapter name is empty or already registered: {name!r}.")
    _FACTORIES[name] = factory


def create_adapter(
    name: str, data: ProcessedData, config: QueryConfig
) -> ModelDataAdapter[Any, Any]:
    """Build a registered adapter or a RecBole adapter (``recbole:Model``).

    Register an exact RecBole name to override its default data preparation.
    RecBole is imported only when a RecBole adapter is requested.
    """
    if name in _FACTORIES:
        return _FACTORIES[name](data, config)
    if name.startswith("recbole:") and name.removeprefix("recbole:"):
        from recbole.utils import ModelType, get_model

        from edurec.datasets.adapters.recbole import RecBoleAdapter

        model = name.removeprefix("recbole:")
        sequential = getattr(get_model(model), "type", None) == ModelType.SEQUENTIAL
        return RecBoleAdapter(data, config, sequential=sequential)
    raise ValueError(
        f"Unknown data adapter {name!r}; register it with register_adapter()."
    )
