from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self, get_type_hints

import yaml

from edurec import settings
from edurec.recsys.archs.modules.interaction_context import ContextConfig
from edurec.recsys.archs.modules.kg_encoder import EdgeType, GraphEncoderConfig
from edurec.recsys.archs.modules.scorer import ScorerConfig
from edurec.recsys.archs.modules.seq_encoder import (
    SeqEncoderConfig,
    TransformerSeqEncoderConfig,
)
from edurec.recsys.archs.modules.user_state import UserStateConfig


@dataclass
class BaseConfig:
    """Base class for all configs with common save/load methods."""

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

    epochs: int = settings.EPOCHS
    lr: float = settings.LR
    batch_size: int = settings.BATCH_SIZE
    patience: int = settings.PATIENCE
    weight_decay: float = settings.WEIGHT_DECAY
    topks: list[int] = field(default_factory=lambda: list(settings.TOP_KS))
    adaptive_k: bool = settings.ADAPTIVE_K
    max_history: int = settings.MAX_HISTORY_LEN
    deduplicate_interactions: bool = settings.DEDUPLICATE_INTERACTIONS


class ModelArch(StrEnum):
    """Available recommendation architectures."""

    KG_RNN = "kg_rnn"
    SASREC_TEXT = "sasrec_text"


@dataclass
class ModelConfig(BaseConfig):
    """Model architecture configuration.

    The ``kg_rnn`` architecture combines an item knowledge graph encoded by a
    relational GNN, an event sequence (item embedding + interaction context +
    time gap) encoded by a GRU/LSTM, a user profile that initialises the
    recurrent state, and an MLP scorer.
    """

    num_users: int
    num_items: int
    num_item_dense_feats: int
    num_item_text_feats: int
    num_user_dense_feats: int = 0
    num_interaction_dense_feats: int = 0
    user_cat_cardinalities: list[int] = field(default_factory=list)
    interaction_cat_cardinalities: list[int] = field(default_factory=list)
    has_history: bool = True
    kg_node_counts: dict[str, int] = field(default_factory=dict)
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
    use_attention_pooling: bool = settings.USE_ATTENTION_POOLING

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
    max_history_len: int = settings.MAX_HISTORY_LEN

    # Scorer defaults
    hidden_dims: list[int] = field(
        default_factory=lambda: [settings.EMB_DIM * 2, settings.EMB_DIM]
    )

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
            self.use_attention_pooling = False
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
    def kg_encoder(self) -> GraphEncoderConfig:
        edge_types: list[EdgeType] = [
            (edge[0], edge[1], edge[2]) for edge in self.kg_edge_types
        ]
        return GraphEncoderConfig(
            num_items=self.num_items,
            emb_dim=self.emb_dim,
            item_feat_dim=self.effective_item_dense_feats,
            num_layers=self.gnn_layers,
            heads=self.gnn_heads,
            dropout=self.gnn_dropout,
            node_counts=dict(self.kg_node_counts),
            edge_types=edge_types,
            graph_mode=self.graph_mode,
        )

    @property
    def interaction_context(self) -> ContextConfig:
        if not self.use_interaction_features:
            return ContextConfig(emb_dim=self.emb_dim)
        return ContextConfig(
            emb_dim=self.emb_dim,
            dense_dim=self.num_interaction_dense_feats,
            cat_cardinalities=list(self.interaction_cat_cardinalities),
        )

    @property
    def user_state(self) -> UserStateConfig:
        if not self.use_user_features:
            return UserStateConfig(
                emb_dim=self.emb_dim,
                hidden_dim=self.rnn_hidden_dim,
                num_layers=self.rnn_layers,
            )
        return UserStateConfig(
            emb_dim=self.emb_dim,
            hidden_dim=self.rnn_hidden_dim,
            num_layers=self.rnn_layers,
            num_dense_feats=self.num_user_dense_feats,
            cat_cardinalities=list(self.user_cat_cardinalities),
            num_users=self.num_users,
            use_id_embedding=self.use_user_id_embedding,
        )

    @property
    def seq_encoder(self) -> SeqEncoderConfig:
        return SeqEncoderConfig(
            emb_dim=self.emb_dim,
            hidden_dim=self.rnn_hidden_dim,
            num_layers=self.rnn_layers,
            dropout=self.rnn_dropout,
            cell_type=self.rnn_type,
            use_attention_pooling=self.use_attention_pooling,
        )

    @property
    def transformer_seq_encoder(self) -> TransformerSeqEncoderConfig:
        return TransformerSeqEncoderConfig(
            emb_dim=self.emb_dim,
            hidden_dim=self.transformer_hidden_dim,
            num_layers=self.transformer_layers,
            num_heads=self.transformer_heads,
            dropout=self.dropout,
            condition_dim=0,
            max_history_len=self.max_history_len,
            use_attention_pooling=self.use_attention_pooling,
            causal=self.arch == ModelArch.SASREC_TEXT,
            use_last_state=self.arch == ModelArch.SASREC_TEXT,
        )

    @property
    def scorer(self) -> ScorerConfig:
        return ScorerConfig(
            emb_dim=self.emb_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        )


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
