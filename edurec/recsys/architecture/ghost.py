from dataclasses import dataclass, field

import torch
from torch import nn

from ... import settings
from .graph_encoder import GraphEncoder, GraphEncoderConfig
from .mlp_encoder import MLPEncoder, MLPEncoderConfig
from .sasrec import SASRecConfig, SASRecEncoder


@dataclass
class GhostConfig:
    num_users: int
    num_items: int
    num_ctx_feats: int
    emb_dim: int = settings.EMB_DIM
    num_user_dense_feats: int = 0
    num_item_dense_feats: int = 0
    num_user_text_feats: int = 0
    num_item_text_feats: int = 0
    user_cat_cardinalities: list[int] = field(default_factory=list)
    item_cat_cardinalities: list[int] = field(default_factory=list)

    # GCL Defaults
    edge_dropout: float = settings.DROP_EDGES_P
    temperature: float = settings.TAU
    loss_reduction: str = settings.LOSS_REDUCTION
    gnn_layers: int = settings.GNN_LAYERS

    # Scorer Defaults
    n_heads: int = settings.NUM_HEADS
    n_blocks: int = settings.NUM_BLOCKS
    ff_dim: int = settings.FF_DIM
    dropout: float = settings.DROPOUT
    use_item_bias: bool = True

    @property
    def gnn(self) -> GraphEncoderConfig:
        return GraphEncoderConfig(
            num_users=self.num_users,
            num_items=self.num_items,
            emb_dim=self.emb_dim,
            num_layers=self.gnn_layers,
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
            num_ctx_feats=self.num_ctx_feats,
        )


class Ghost(nn.Module):
    def __init__(self, cfg: GhostConfig):
        super().__init__()
        self.cfg = cfg

        self.gnn = GraphEncoder(cfg.gnn)
        self.user_encoder = MLPEncoder(cfg.user_encoder)
        self.item_encoder = MLPEncoder(cfg.item_encoder)
        self.text_projection = (
            nn.Linear(cfg.num_item_text_feats, cfg.emb_dim, bias=False)
            if cfg.num_item_text_feats > 0
            else None
        )
        self.item_content_norm = nn.LayerNorm(cfg.emb_dim)
        self.item_norm = nn.LayerNorm(cfg.emb_dim)
        self.user_norm = nn.LayerNorm(cfg.emb_dim)
        self.item_bias = (
            nn.Parameter(torch.zeros(cfg.num_items)) if cfg.use_item_bias else None
        )
        self.sequence_encoder = SASRecEncoder(cfg.sasrec)

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
        user_graph_embs, item_graph_embs = self.gnn(
            edge_index, u_static_feats, i_static_feats
        )
        user_feature_embs = self.user_encoder(u_static_feats)
        item_feature_embs = self._encode_item_features(i_static_feats)

        item_emb = self.item_norm(item_graph_embs + item_feature_embs)
        padded_item_embs = torch.cat(
            [item_emb.new_zeros(1, item_emb.size(1)), item_emb], dim=0
        )

        hist_emb = padded_item_embs[h_ids.clamp(min=0)]
        seq_user_emb = self.sequence_encoder(hist_emb, h_mask, h_ctx)

        user_graph_emb = user_graph_embs[u_ids]
        user_feature_emb = user_feature_embs[u_ids]
        user_emb = self.user_norm(user_graph_emb + user_feature_emb + seq_user_emb)

        scores = scores = user_emb @ item_emb.T

        if self.item_bias is not None:
            scores = scores + self.item_bias

        return scores

    def _encode_item_features(self, item_static_feats: torch.Tensor) -> torch.Tensor:
        struct_dim = max(
            self.cfg.num_item_dense_feats - self.cfg.num_item_text_feats, 0
        )
        cat_start = self.cfg.num_item_dense_feats

        parts: list[torch.Tensor] = []
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
            return item_feature_emb

        text_start = struct_dim
        text_end = text_start + self.cfg.num_item_text_feats
        text_emb = self.text_projection(item_static_feats[..., text_start:text_end])
        return self.item_content_norm(item_feature_emb + text_emb)
