from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Self

import torch
import yaml
from torch import nn

from ... import settings
from .fusion import FusionConfig, MaskedGatedFusion, SumFusion
from .graph_encoder import GraphEncoder, GraphEncoderConfig
from .mlp_encoder import MLPEncoder, MLPEncoderConfig
from .scorer import Scorer, ScorerConfig
from .seq_encoder import SeqEncoder, SeqEncoderConfig


@dataclass
class EDuRecConfig:
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

    # Training Defaults
    lr: float = settings.LR
    weight_decay: float = settings.WEIGHT_DECAY
    topks: list[int] = field(default_factory=lambda: settings.TOP_KS)
    alpha: float = settings.LOSS_ALPHA
    adaptive_k: bool = settings.ADAPTIVE_K

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
    def fusion(self) -> FusionConfig:
        return FusionConfig(
            emb_dim=self.emb_dim,
            num_sources=3,
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


class EDuRec(nn.Module):
    def __init__(self, cfg: EDuRecConfig):
        super().__init__()
        self.cfg = cfg

        available = cfg.available_modules
        self.available_modules = available

        self.gnn = GraphEncoder(cfg.graph_encoder) if available["graph"] else None
        self.user_encoder = (
            MLPEncoder(cfg.user_encoder) if available["user_features"] else None
        )
        self.item_encoder = (
            MLPEncoder(cfg.item_encoder) if available["item_features"] else None
        )
        self.context_encoder = (
            MLPEncoder(cfg.context_encoder) if available["context"] else None
        )
        item_sources = int(available["graph"]) + int(available["item_features"])
        user_sources = (
            int(available["graph"])
            + int(available["user_features"])
            + int(available["sequence"])
        )
        if item_sources == 0 or user_sources == 0:
            raise ValueError(
                "The effective configuration must provide at least one user and "
                "one item representation module."
            )
        self.item_fusion = self._make_fusion(item_sources)
        self.user_fusion = self._make_fusion(user_sources)
        self.item_bias = (
            nn.Parameter(torch.zeros(cfg.num_items)) if cfg.use_item_bias else None
        )
        self.sequence_encoder = (
            SeqEncoder(cfg.seq_encoder) if available["sequence"] else None
        )
        self.scorer = Scorer(cfg.scorer)

    def _make_fusion(
        self, num_sources: int
    ) -> MaskedGatedFusion | SumFusion | None:
        # A single source needs neither gates nor normalization parameters.
        if num_sources == 1:
            return None
        fusion_cfg = FusionConfig(
            emb_dim=self.cfg.emb_dim,
            num_sources=num_sources,
            dropout=self.cfg.dropout,
        )
        if self.cfg.fusion_type == "sum":
            return SumFusion(fusion_cfg)
        if self.cfg.fusion_type == "masked_gated":
            return MaskedGatedFusion(fusion_cfg)
        raise ValueError(f"Unknown fusion type: {self.cfg.fusion_type!r}.")

    @staticmethod
    def _fuse(
        sources: list[torch.Tensor],
        fusion: MaskedGatedFusion | SumFusion | None,
    ) -> torch.Tensor:
        return sources[0] if fusion is None else fusion(sources)

    def forward(
        self,
        u_ids: torch.Tensor,
        h_ids: torch.Tensor,
        h_mask: torch.Tensor,
        edge_index: torch.Tensor,
        u_static_feats: torch.Tensor,
        i_static_feats: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        user_sources = []
        item_sources = []

        if self.gnn is not None:
            user_graph, item_graph = self.gnn(edge_index)
            user_sources.append(user_graph[u_ids])
            item_sources.append(item_graph)

        if self.user_encoder is not None:
            user_sources.append(self.user_encoder(u_static_feats)[u_ids])
        if self.item_encoder is not None:
            item_sources.append(self.item_encoder(i_static_feats))
        context_emb = None
        if self.context_encoder is not None:
            if context is None:
                raise ValueError(
                    "context is required when the context module is active."
                )
            context_emb = self.context_encoder(context)

        item_emb = self._fuse(item_sources, self.item_fusion)

        if self.sequence_encoder is not None:
            padded = torch.cat([item_emb.new_zeros(1, item_emb.size(1)), item_emb])
            hist = padded[h_ids.clamp(min=0)]
            seq_user = self.sequence_encoder(hist, h_mask)
            user_sources.append(seq_user)

        user_emb = self._fuse(user_sources, self.user_fusion)

        scores = self.scorer(user_emb, item_emb, context_emb)

        if self.item_bias is not None:
            scores = scores + self.item_bias

        return scores
