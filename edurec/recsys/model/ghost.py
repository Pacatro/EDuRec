from dataclasses import dataclass, field

import torch
from torch import nn
from torch_geometric.data import Data

from ... import config
from .gnn_encoder import GnnEncoder, GnnEncoderConfig, LossReduction
from .ranker import Ranker, RankerConfig


@dataclass
class GhostConfig:
    num_users: int
    num_items: int
    num_ctx_feats: int
    emb_dim: int = config.EMB_DIM
    num_user_numeric_feats: int = 0
    num_item_numeric_feats: int = 0
    user_cat_cardinalities: list[int] = field(default_factory=list)
    item_cat_cardinalities: list[int] = field(default_factory=list)

    # GCL Defaults
    edge_dropout: float = config.DROP_EDGES_P
    temperature: float = config.TAU
    loss_reduction: str = config.LOSS_REDUCTION
    gnn_layers: int = config.GNN_LAYERS

    # Ranker Defaults
    n_heads: int = config.NUM_HEADS
    n_blocks: int = config.NUM_BLOCKS
    ff_dim: int = config.FF_DIM
    dropout: float = config.DROPOUT
    num_scores: int = config.NUM_SCORES

    @property
    def gnn(self) -> GnnEncoderConfig:
        return GnnEncoderConfig(
            num_users=self.num_users,
            num_items=self.num_items,
            emb_dim=self.emb_dim,
            drop_edges_p=self.edge_dropout,
            tau=self.temperature,
            loss_reduc=LossReduction(self.loss_reduction),
            num_layers=self.gnn_layers,
        )

    @property
    def ranker(self) -> RankerConfig:
        return RankerConfig(
            emb_dim=self.emb_dim,
            n_heads=self.n_heads,
            n_blocks=self.n_blocks,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            num_scores=self.num_scores,
        )


class Ghost(nn.Module):
    def __init__(self, cfg: GhostConfig):
        super().__init__()
        self.cfg = cfg

        self.ctx_proj = (
            nn.Linear(cfg.num_ctx_feats, cfg.emb_dim, bias=False)
            if cfg.num_ctx_feats > 0
            else None
        )

        self.user_static_encoder = StaticFeatureEncoder(
            cfg.num_user_numeric_feats,
            cfg.user_cat_cardinalities,
            cfg.emb_dim,
        )
        self.item_static_encoder = StaticFeatureEncoder(
            cfg.num_item_numeric_feats,
            cfg.item_cat_cardinalities,
            cfg.emb_dim,
        )
        self.norm = nn.LayerNorm(cfg.emb_dim)
        self.gnn = GnnEncoder(cfg.gnn)
        self.ranker = Ranker(cfg.ranker)

        self.register_buffer("edge_index", None, persistent=False)

    def forward(
        self,
        u_ids: torch.Tensor,  # [B]
        h_ids: torch.Tensor,  # [B, H]
        h_ctx: torch.Tensor,  # [B, H, num_ctx_feats]
        h_mask: torch.Tensor,  # [B, H]
        c_ids: torch.Tensor,  # [B, C]
        inter_graph: Data,
        u_static_global: torch.Tensor,  # [B, num_users_feats]
        i_static_global: torch.Tensor,  # [B, num_items_feats]
    ) -> torch.Tensor:
        self.edge_index = inter_graph.edge_index

        user_embs, item_embs = self.gnn(inter_graph)
        padded_item_embs = torch.cat(
            [item_embs.new_zeros(1, item_embs.size(1)), item_embs], dim=0
        )
        padded_item_static = self._pad_item_static_features(i_static_global)

        user_emb = user_embs[u_ids]  # [B, D]
        hist_emb = self._build_hist_emb(
            padded_item_embs, h_ids, h_ctx, h_mask
        )  # [B, H, D]
        candidate_emb = padded_item_embs[c_ids]  # [B, C, D]

        user_feats = self.user_static_encoder(u_static_global[u_ids])  # [B, D]
        hist_feats = self.item_static_encoder(padded_item_static[h_ids])  # [B, H, D]
        candidate_feats = self.item_static_encoder(
            padded_item_static[c_ids]
        )  # [B, C, D]

        user_emb = self.norm(user_emb + user_feats)
        hist_emb = self.norm(hist_emb + hist_feats)
        hist_emb = hist_emb * h_mask.unsqueeze(-1).float()
        candidate_emb = self.norm(candidate_emb + candidate_feats)

        scores = self.ranker(
            user_emb, hist_emb, candidate_emb, h_mask
        )  # [B, C, num_scores]

        return scores

    def _build_hist_emb(
        self,
        item_embs: torch.Tensor,
        history_ids: torch.Tensor,
        history_ctx: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        history_ids = history_ids.clamp(min=0)
        hist_emb = item_embs[history_ids]  # [B, S, D]
        hist_emb = hist_emb * history_mask.unsqueeze(-1).float()  # padding → 0

        if self.ctx_proj is not None:
            ctx_emb = self.ctx_proj(history_ctx)  # [B, S, D]
            ctx_emb = ctx_emb * history_mask.unsqueeze(-1).float()
            hist_emb = hist_emb + ctx_emb

        return hist_emb

    def _pad_item_static_features(self, item_static: torch.Tensor) -> torch.Tensor:
        if item_static.ndim != 2:
            raise RuntimeError("Item static features must be a 2D tensor.")

        pad_row = item_static.new_zeros((1, item_static.size(1)))
        num_numeric = self.cfg.num_item_numeric_feats
        num_cats = len(self.cfg.item_cat_cardinalities)

        if num_cats > 0:
            pad_row[:, num_numeric : num_numeric + num_cats] = -1

        return torch.cat([pad_row, item_static], dim=0)


class StaticFeatureEncoder(nn.Module):
    def __init__(
        self,
        num_numeric_features: int,
        categorical_cardinalities: list[int],
        emb_dim: int,
    ):
        super().__init__()
        self.num_numeric_features = num_numeric_features
        self.num_categorical_features = len(categorical_cardinalities)
        self.emb_dim = emb_dim

        self.numeric_proj = (
            nn.Linear(num_numeric_features, emb_dim, bias=False)
            if num_numeric_features > 0
            else None
        )
        self.cat_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, emb_dim, padding_idx=0)
            for cardinality in categorical_cardinalities
        )
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.new_zeros((*x.shape[:-1], self.emb_dim))

        if self.numeric_proj is not None:
            numeric_feats = x[..., : self.num_numeric_features].float()
            out = out + self.numeric_proj(numeric_feats)

        if self.cat_embeddings:
            cat_feats = x[..., self.num_numeric_features :].long() + 1
            cat_feats = cat_feats.clamp(min=0)
            for idx, embedding in enumerate(self.cat_embeddings):
                out = out + embedding(cat_feats[..., idx])

        return self.norm(out)
