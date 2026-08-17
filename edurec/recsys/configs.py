from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Self

import yaml

from .. import settings
from .architecture.graph_encoder import GraphEncoderConfig
from .architecture.mlp_encoder import MLPEncoderConfig
from .architecture.scorer import ScorerConfig
from .architecture.seq_encoder import SeqEncoderConfig


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
            return cls(**yaml.safe_load(f))


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
    """Model architecture configuration for EDuRec."""

    num_users: int
    num_items: int
    num_ctx_feats: int
    num_user_dense_feats: int
    num_item_dense_feats: int
    num_user_text_feats: int
    num_item_text_feats: int
    user_cat_cardinalities: list[int]
    item_cat_cardinalities: list[int]
    has_history: bool = True
    emb_dim: int = settings.EMB_DIM
    use_item_bias: bool = True
    dropout: float = settings.DROPOUT

    # Ablations
    graph_mode: Literal["id", "lightgcn", "none"] = "lightgcn"
    use_user_features: bool = True
    use_item_features: bool = True
    use_text_features: bool = True
    use_seq_encoder: bool = True
    use_context: bool = True
    use_gcl: bool = True
    scorer_type: Literal["mlp", "dot"] = "mlp"
    fusion_type: Literal["masked_gated", "sum"] = "masked_gated"

    # GCL Defaults
    edge_dropout: float = settings.DROP_EDGES_P
    temperature: float = settings.TAU
    loss_reduction: str = settings.LOSS_REDUCTION
    gnn_layers: int = settings.GNN_LAYERS

    # SASRec Defaults
    n_heads: int = settings.NUM_HEADS
    n_blocks: int = settings.NUM_BLOCKS
    ff_dim: int = settings.FF_DIM

    # Scorer defaults
    hidden_dims: list[int] = field(
        default_factory=lambda: [settings.EMB_DIM * 2, settings.EMB_DIM]
    )

    @property
    def has_user_features(self) -> bool:
        """Whether the dataset can feed the user feature encoder."""
        return self.num_user_dense_feats > 0 or bool(self.user_cat_cardinalities)

    @property
    def has_item_features(self) -> bool:
        """Whether the dataset can feed the item feature encoder."""
        return self.num_item_dense_feats > 0 or bool(self.item_cat_cardinalities)

    @property
    def available_modules(self) -> dict[str, bool]:
        """Effective modules after combining dataset availability and ablations."""
        graph = self.graph_mode != "none"
        sequence = self.use_seq_encoder and self.has_history
        return {
            "graph": graph,
            "user_features": self.use_user_features and self.has_user_features,
            "item_features": self.use_item_features and self.has_item_features,
            "sequence": sequence,
            "context": self.use_context and self.num_ctx_feats > 0,
        }

    @property
    def graph_encoder(self) -> GraphEncoderConfig:
        return GraphEncoderConfig(
            num_users=self.num_users,
            num_items=self.num_items,
            emb_dim=self.emb_dim,
            num_layers=self.gnn_layers if self.graph_mode == "lightgcn" else 0,
            num_user_dense_feats=self.num_user_dense_feats,
            num_item_dense_feats=self.num_item_dense_feats,
            user_cat_cardinalities=self.user_cat_cardinalities,
            item_cat_cardinalities=self.item_cat_cardinalities,
        )

    @property
    def user_encoder(self) -> MLPEncoderConfig:
        return MLPEncoderConfig(
            num_dense_features=self.num_user_dense_feats,
            categorical_cardinalities=self.user_cat_cardinalities,
            output_dim=self.emb_dim,
            dropout=self.dropout,
        )

    @property
    def item_encoder(self) -> MLPEncoderConfig:
        return MLPEncoderConfig(
            num_dense_features=self.num_item_dense_feats,
            categorical_cardinalities=self.item_cat_cardinalities,
            output_dim=self.emb_dim,
            dropout=self.dropout,
        )

    @property
    def context_encoder(self) -> MLPEncoderConfig:
        return MLPEncoderConfig(
            num_dense_features=self.num_ctx_feats,
            output_dim=self.emb_dim,
            dropout=self.dropout,
        )

    @property
    def seq_encoder(self) -> SeqEncoderConfig:
        return SeqEncoderConfig(
            emb_dim=self.emb_dim,
            n_heads=self.n_heads,
            n_blocks=self.n_blocks,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
        )

    @property
    def scorer(self) -> ScorerConfig:
        return ScorerConfig(
            emb_dim=self.emb_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            scorer_type=self.scorer_type,
            use_context=self.available_modules["context"],
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
