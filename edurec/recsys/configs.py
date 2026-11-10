from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Literal, Self, get_type_hints

import yaml

from edurec import settings


@dataclass
class BaseConfig:
    """Base class for all configs with common save/load methods."""

    section: ClassVar[str] = ""
    top_level_fields: ClassVar[frozenset[str]] = frozenset()

    @staticmethod
    def _coerce(value: Any, annotation: Any) -> Any:
        """Coerce YAML scalars to the field's declared numeric type.

        PyYAML follows the YAML 1.1 spec, which requires a dot in a float
        mantissa. Literals like ``1e-5`` are therefore loaded as strings, which
        would otherwise only fail later inside the optimizer.
        """
        if isinstance(value, str) and annotation in (float, int):
            return annotation(value)
        return value

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            key: value.value if isinstance(value, StrEnum) else value
            for key, value in asdict(self).items()
        }
        with open(path, "w") as f:
            yaml.safe_dump(payload, f)

    @classmethod
    def load(cls, path: Path | str) -> Self:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            payload = yaml.safe_load(f) or {}

        if not isinstance(payload, Mapping):
            raise TypeError("Config must be a mapping.")
        if cls.section and ("model" in payload or "train" in payload):
            values = payload.get(cls.section, {})
            if not isinstance(values, Mapping):
                raise ValueError(
                    f"Config section '{cls.section}' must be a mapping."
                )
            document = payload
            payload = dict(values)
            for name in cls.top_level_fields:
                if name in document:
                    payload[name] = document[name]

        # Ignore fields left over from older config schemas instead of failing.
        known = {item.name for item in fields(cls)}
        hints = get_type_hints(cls)
        return cls(
            **{
                key: cls._coerce(value, hints.get(key))
                for key, value in payload.items()
                if key in known
            }
        )


@dataclass
class TrainConfig(BaseConfig):
    """Training hyperparameters, independent from the model architecture."""

    section: ClassVar[str] = "train"

    epochs: int = settings.EPOCHS
    lr: float = settings.LR
    batch_size: int = settings.BATCH_SIZE
    patience: int = settings.PATIENCE
    weight_decay: float = settings.WEIGHT_DECAY
    topks: list[int] = field(default_factory=lambda: list(settings.TOP_KS))
    adaptive_k: bool = settings.ADAPTIVE_K
    max_history: int = settings.MAX_HISTORY_LEN


class ModelArch(StrEnum):
    """Available recommendation architectures."""

    KG_RNN = "kg_rnn"
    SASREC_TEXT = "sasrec_text"


#: Model fields computed from the processed dataset. ``build_config`` fills
#: them at runtime, so they must not be persisted in configuration files.
DATASET_MODEL_FIELDS: frozenset[str] = frozenset(
    {
        "num_users",
        "num_items",
        "num_item_dense_feats",
        "num_item_text_feats",
        "num_user_dense_feats",
        "num_interaction_dense_feats",
        "user_cat_cardinalities",
        "interaction_cat_cardinalities",
        "has_history",
        "kg_node_counts",
        "kg_node_feat_dims",
        "kg_edge_types",
    }
)


@dataclass
class ModelConfig(BaseConfig):
    """Model architecture configuration.

    The ``kg_rnn`` architecture combines an item knowledge graph encoded by a
    relational GNN, an event sequence (item embedding + interaction context +
    time gap) encoded by a GRU/LSTM, a user profile that initialises the
    recurrent state, and an MLP scorer.

    Dataset dimensions default to zero so that saved architecture files can
    omit them. Training and evaluation derive their real values from the
    processed data before constructing the model.
    """

    section: ClassVar[str] = "model"
    top_level_fields: ClassVar[frozenset[str]] = frozenset({"arch"})

    num_users: int = 0
    num_items: int = 0
    num_item_dense_feats: int = 0
    num_item_text_feats: int = 0
    num_user_dense_feats: int = 0
    num_interaction_dense_feats: int = 0
    user_cat_cardinalities: list[int] = field(default_factory=list)
    interaction_cat_cardinalities: list[int] = field(default_factory=list)
    has_history: bool = True
    kg_node_counts: dict[str, int] = field(default_factory=dict)
    kg_node_feat_dims: dict[str, int] = field(default_factory=dict)
    kg_edge_types: list[list[str]] = field(default_factory=list)
    emb_dim: int = settings.EMB_DIM
    use_item_bias: bool = True
    dropout: float = settings.DROPOUT

    # Architecture selection
    arch: ModelArch = ModelArch.KG_RNN

    # Ablations / feature toggles
    graph_mode: Literal["kg", "id"] = "kg"
    use_text_features: bool = True
    use_item_features: bool = settings.USE_ITEM_FEATURES
    use_user_features: bool = True
    use_interaction_features: bool = settings.USE_INTERACTION_FEATURES
    use_time_features: bool = settings.USE_TIME_FEATURES

    # User profile
    use_user_id_embedding: bool = False

    # GNN defaults
    gnn_layers: int = settings.GNN_LAYERS
    gnn_heads: int = settings.GNN_HEADS
    gnn_dropout: float = settings.DROPOUT

    # Training-only graph contrastive regularization (zero preserves baselines).
    use_gcl: bool = True
    gcl_weight: float = 0.0
    gcl_temperature: float = 0.2
    gcl_edge_dropout: float = 0.1
    gcl_max_items: int = 512

    # Recurrent sequence defaults
    rnn_type: Literal["gru", "lstm"] = settings.SEQ_CELL
    rnn_hidden_dim: int = settings.GRU_HIDDEN_DIM
    rnn_layers: int = settings.GRU_LAYERS
    rnn_dropout: float = settings.DROPOUT

    # Transformer defaults (SASRecText)
    transformer_hidden_dim: int = settings.TRANSFORMER_HIDDEN_DIM
    transformer_layers: int = settings.TRANSFORMER_LAYERS
    transformer_heads: int = settings.TRANSFORMER_HEADS

    # Scorer defaults
    scorer_max_pairs_per_chunk: int = 32768
    hidden_dims: list[int] = field(
        default_factory=lambda: [settings.EMB_DIM * 2, settings.EMB_DIM]
    )

    def save(self, path: Path | str) -> None:
        """Save architecture settings without dataset-derived dimensions."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(model_config_payload(self), f)

    def __post_init__(self) -> None:
        if not 0.0 <= self.gcl_weight < float("inf"):
            raise ValueError("gcl_weight must be finite and non-negative.")
        if not 0.0 < self.gcl_temperature < float("inf"):
            raise ValueError("gcl_temperature must be finite and positive.")
        if not 0.0 <= self.gcl_edge_dropout <= 1.0:
            raise ValueError("gcl_edge_dropout must be between 0 and 1.")
        if self.gcl_max_items < 2:
            raise ValueError("gcl_max_items must be at least 2.")
        if self.gcl_enabled and self.arch != ModelArch.KG_RNN:
            raise ValueError("GCL is only supported by the kg_rnn architecture.")
        if self.arch == ModelArch.SASREC_TEXT:
            # These modules are fixed by the SASRec + content architecture.
            self.graph_mode = "id"
            self.use_item_features = True
            self.use_user_features = False
            self.use_user_id_embedding = False
            self.use_interaction_features = False
            self.use_time_features = False
            self.use_item_bias = False

    @property
    def gcl_enabled(self) -> bool:
        """Whether GCL runs; disabling it preserves its tuned hyperparameters."""
        return self.use_gcl and self.gcl_weight > 0.0

    @property
    def effective_item_dense_feats(self) -> int:
        """Item dense features actually fed to the graph after ablations."""
        if not self.use_item_features:
            return 0
        if self.use_text_features:
            return self.num_item_dense_feats
        return self.num_item_dense_feats - self.num_item_text_feats

    @property
    def has_user_profile(self) -> bool:
        """Whether any static user signal is available to condition on."""
        return self.use_user_features and (
            self.num_user_dense_feats > 0
            or len(self.user_cat_cardinalities) > 0
            or self.use_user_id_embedding
        )

    @property
    def has_interaction_context(self) -> bool:
        """Whether any per-interaction context signal is available."""
        return self.use_interaction_features and (
            self.num_interaction_dense_feats > 0
            or len(self.interaction_cat_cardinalities) > 0
        )


def monitor_mode(monitor: str) -> Literal["min", "max"]:
    """Whether the monitored metric improves by decreasing or increasing."""
    return "min" if monitor.lower().endswith("loss") else "max"


def resolve_train_config(
    cli: Mapping[str, Any] | None = None,
    saved_path: Path | str | None = None,
    defaults: TrainConfig | None = None,
) -> TrainConfig:
    """Resolve the effective training config for a run.

    Precedence: explicit CLI values win over the saved config file, which
    wins over the provided defaults (global or per-dataset).
    """
    resolved = defaults if defaults is not None else TrainConfig()
    if saved_path is not None and Path(saved_path).exists():
        resolved = replace(resolved, **asdict(TrainConfig.load(saved_path)))
    if cli:
        resolved = replace(
            resolved,
            **{name: value for name, value in cli.items() if value is not None},
        )
    return resolved


def monitor_topk(top_k: int | None, train_cfg: TrainConfig) -> int:
    """The cutoff that drives early stopping and checkpointing."""
    return top_k if top_k is not None else max(train_cfg.topks)


def model_config_payload(config: ModelConfig) -> dict[str, Any]:
    """Plain model configuration without dataset-derived values.

    Dataset dimensions, cardinalities and graph structure are rebuilt from the
    processed data by ``build_config``. Keeping them out of YAML/JSON files
    prevents stale values after rebuilding or sampling a different catalog.
    """
    payload = asdict(config)
    for name in DATASET_MODEL_FIELDS:
        payload.pop(name, None)
    payload["arch"] = config.arch.value
    return payload


def save_run_config(path: Path | str, model: ModelConfig, train: TrainConfig) -> None:
    """Save model and training settings without dataset-derived model values."""
    model_payload = model_config_payload(model)
    arch = model_payload.pop("arch")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {"arch": arch, "model": model_payload, "train": asdict(train)},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
