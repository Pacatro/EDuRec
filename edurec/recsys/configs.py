from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal, Self

import yaml

from .. import settings
from .archs.modules.kg_encoder import EdgeType, KGEncoderConfig
from .archs.modules.scorer import ScorerConfig
from .archs.modules.seq_encoder import SeqEncoderConfig


@dataclass
class BaseConfig:
    """Base class for all configs with common save/load methods."""

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f)

    @classmethod
    def load(cls, path: Path | str) -> Self:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            payload = yaml.safe_load(f) or {}

        # Ignore fields left over from older config schemas instead of failing.
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in known})


@dataclass
class TrainConfig(BaseConfig):
    """Training hyperparameters, independent from the model architecture."""

    epochs: int = settings.EPOCHS
    lr: float = settings.LR
    batch_size: int = settings.BATCH_SIZE
    patience: int = settings.PATIENCE
    weight_decay: float = settings.WEIGHT_DECAY
    topks: list[int] = field(default_factory=lambda: list(settings.TOP_KS))
    alpha: float = settings.LOSS_ALPHA
    adaptive_k: bool = settings.ADAPTIVE_K


@dataclass
class ModelConfig(BaseConfig):
    """Model architecture configuration for KGRNN."""

    num_users: int
    num_items: int
    num_user_dense_feats: int
    num_item_dense_feats: int
    num_user_text_feats: int
    num_item_text_feats: int
    has_history: bool = True
    kg_node_counts: dict[str, int] = field(default_factory=dict)
    kg_edge_types: list[list[str]] = field(default_factory=list)
    emb_dim: int = settings.EMB_DIM
    use_item_bias: bool = True
    dropout: float = settings.DROPOUT

    # Ablations
    graph_mode: Literal["kg", "id"] = "kg"
    use_text_features: bool = True
    use_seq_encoder: bool = True
    use_gcl: bool = True
    scorer_type: Literal["mlp", "dot"] = "mlp"

    # GCL Defaults
    edge_dropout: float = settings.DROP_EDGES_P
    temperature: float = settings.TAU
    loss_reduction: str = settings.LOSS_REDUCTION
    gnn_layers: int = settings.GNN_LAYERS

    # GRU Defaults
    gru_hidden_dim: int = settings.GRU_HIDDEN_DIM
    gru_layers: int = settings.GRU_LAYERS
    seq_cell: Literal["gru", "lstm"] = settings.SEQ_CELL

    # Scorer defaults
    hidden_dims: list[int] = field(
        default_factory=lambda: [settings.EMB_DIM * 2, settings.EMB_DIM]
    )

    @property
    def effective_user_dense_feats(self) -> int:
        """User dense features actually fed to the graph after ablations."""
        if self.use_text_features:
            return self.num_user_dense_feats
        return self.num_user_dense_feats - self.num_user_text_feats

    @property
    def effective_item_dense_feats(self) -> int:
        """Item dense features actually fed to the graph after ablations."""
        if self.use_text_features:
            return self.num_item_dense_feats
        return self.num_item_dense_feats - self.num_item_text_feats

    @property
    def available_modules(self) -> dict[str, bool]:
        """Effective modules after combining dataset availability and ablations."""
        return {
            "graph": self.graph_mode in {"kg", "id"},
            "sequence": self.use_seq_encoder and self.has_history,
        }

    @property
    def kg_encoder(self) -> KGEncoderConfig:
        use_kg = self.graph_mode == "kg"
        edge_types: list[EdgeType] = (
            [(edge[0], edge[1], edge[2]) for edge in self.kg_edge_types]
            if use_kg
            else []
        )
        return KGEncoderConfig(
            num_users=self.num_users,
            num_items=self.num_items,
            emb_dim=self.emb_dim,
            user_feat_dim=self.effective_user_dense_feats,
            item_feat_dim=self.effective_item_dense_feats,
            num_layers=self.gnn_layers,
            node_counts=dict(self.kg_node_counts) if use_kg else {},
            edge_types=edge_types,
            graph_mode=self.graph_mode,
        )

    @property
    def seq_encoder(self) -> SeqEncoderConfig:
        return SeqEncoderConfig(
            emb_dim=self.emb_dim,
            hidden_dim=self.gru_hidden_dim,
            num_layers=self.gru_layers,
            dropout=self.dropout,
            cell_type=self.seq_cell,
        )

    @property
    def scorer(self) -> ScorerConfig:
        return ScorerConfig(
            emb_dim=self.emb_dim,
            num_user_sources=1 + int(self.available_modules["sequence"]),
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            scorer_type=self.scorer_type,
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
