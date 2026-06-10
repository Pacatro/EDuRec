from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Self
import yaml

import torch
from torch import nn

from ... import settings
from .graph_encoder import GraphEncoder, GraphEncoderConfig
from .mlp_encoder import MLPEncoder, MLPEncoderConfig
from .sasrec import SASRecConfig, SASRecEncoder
from .scorer import Scorer, ScorerConfig


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
    emb_dim: int = settings.EMB_DIM
    use_item_bias: bool = True
    dropout: float = settings.DROPOUT
    graph_mode: Literal["id", "lightgcn", "none"] = "id"
    use_user_features: bool = True
    use_item_features: bool = True
    use_text_features: bool = True
    use_sasrec: bool = False
    use_context: bool = False
    use_gcl: bool = False
    scorer_type: Literal["mlp", "dot"] = "mlp"

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
    def gnn(self) -> GraphEncoderConfig:
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
        struct_dense_feats = max(
            self.num_item_dense_feats - self.num_item_text_feats, 0
        )
        return MLPEncoderConfig(
            num_dense_features=struct_dense_feats,
            categorical_cardinalities=self.item_cat_cardinalities,
            output_dim=self.emb_dim,
            dropout=self.dropout,
        )

    @property
    def sasrec(self) -> SASRecConfig:
        return SASRecConfig(
            emb_dim=self.emb_dim,
            n_heads=self.n_heads,
            n_blocks=self.n_blocks,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            num_ctx_feats=self.num_ctx_feats if self.use_context else 0,
        )

    @property
    def scorer(self) -> ScorerConfig:
        return ScorerConfig(
            emb_dim=self.emb_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            scorer_type=self.scorer_type,
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

        self.gnn = GraphEncoder(cfg.gnn) if cfg.graph_mode != "none" else None
        self.user_encoder = (
            MLPEncoder(cfg.user_encoder) if cfg.use_user_features else None
        )
        self.item_encoder = (
            MLPEncoder(cfg.item_encoder) if cfg.use_item_features else None
        )
        self.text_projection = (
            nn.Linear(cfg.num_item_text_feats, cfg.emb_dim, bias=False)
            if cfg.use_text_features and cfg.num_item_text_feats > 0
            else None
        )
        self.item_norm = nn.LayerNorm(cfg.emb_dim)
        self.user_norm = nn.LayerNorm(cfg.emb_dim)
        self.item_bias = (
            nn.Parameter(torch.zeros(cfg.num_items)) if cfg.use_item_bias else None
        )
        self.sequence_encoder = SASRecEncoder(cfg.sasrec) if cfg.use_sasrec else None
        self.scorer = Scorer(cfg.scorer)

    def forward(
        self,
        u_ids: torch.Tensor,
        h_ids: torch.Tensor,
        h_ctx: torch.Tensor,
        h_mask: torch.Tensor,
        edge_index: torch.Tensor,
        u_static_feats: torch.Tensor,
        i_static_feats: torch.Tensor,
    ) -> torch.Tensor:
        if self.gnn is None:
            user_graph_embs = u_static_feats.new_zeros(
                self.cfg.num_users, self.cfg.emb_dim
            )
            item_graph_embs = i_static_feats.new_zeros(
                self.cfg.num_items, self.cfg.emb_dim
            )
        else:
            user_graph_embs, item_graph_embs = self.gnn(edge_index)

        if self.user_encoder is None:
            user_feature_embs = u_static_feats.new_zeros(
                self.cfg.num_users, self.cfg.emb_dim
            )
        else:
            user_feature_embs = self.user_encoder(u_static_feats)
        item_feature_embs, text_embs = self._encode_item_features(i_static_feats)

        item_feats_emb = (
            item_graph_embs + item_feature_embs
            if text_embs is None
            else item_graph_embs + item_feature_embs + text_embs
        )

        item_emb = self.item_norm(item_feats_emb)
        padded_item_embs = torch.cat(
            [item_emb.new_zeros(1, item_emb.size(1)), item_emb], dim=0
        )

        hist_emb = padded_item_embs[h_ids.clamp(min=0)]
        if self.sequence_encoder is None:
            seq_user_emb = hist_emb.new_zeros(hist_emb.size(0), self.cfg.emb_dim)
        else:
            seq_user_emb = self.sequence_encoder(
                hist_emb,
                h_mask,
                h_ctx if self.cfg.use_context else None,
            )

        user_graph_emb = user_graph_embs[u_ids]
        user_feature_emb = user_feature_embs[u_ids]
        user_emb = self.user_norm(user_graph_emb + user_feature_emb + seq_user_emb)

        scores = self.scorer(user_emb, item_emb)

        if self.item_bias is not None:
            scores = scores + self.item_bias

        return scores

    def _encode_item_features(
        self, item_static_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        struct_dim = max(
            self.cfg.num_item_dense_feats - self.cfg.num_item_text_feats, 0
        )
        cat_start = self.cfg.num_item_dense_feats

        if self.item_encoder is None:
            item_feature_emb = item_static_feats.new_zeros(
                item_static_feats.size(0), self.cfg.emb_dim
            )
        else:
            parts = []
            if struct_dim > 0:
                parts.append(item_static_feats[..., :struct_dim])
            if self.cfg.item_cat_cardinalities:
                parts.append(item_static_feats[..., cat_start:])

            if parts:
                structured_inputs = torch.cat(parts, dim=-1)
                item_feature_emb = self.item_encoder(structured_inputs)
            else:
                item_feature_emb = item_static_feats.new_zeros(
                    item_static_feats.size(0), self.cfg.emb_dim
                )

        if self.text_projection is None:
            return item_feature_emb, None

        text_start = struct_dim
        text_end = text_start + self.cfg.num_item_text_feats
        text_emb = self.text_projection(item_static_feats[..., text_start:text_end])
        return item_feature_emb, text_emb
