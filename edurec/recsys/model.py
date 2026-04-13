from dataclasses import dataclass, field

import torch
from torch import nn
from torch_geometric.data import Data

from .. import config
from .gnn_encoder import GnnEncoder, GnnEncoderConfig, LossReduction
from .ranker import Ranker, RankerConfig


@dataclass
class GnnRankerConfig:
    num_users: int
    num_items: int
    num_ctx_feats: int
    emb_dim: int = config.EMB_DIM
    num_user_dense_feats: int = 0
    num_item_dense_feats: int = 0
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
            num_user_dense_feats=self.num_user_dense_feats,
            num_item_dense_feats=self.num_item_dense_feats,
            user_cat_cardinalities=self.user_cat_cardinalities,
            item_cat_cardinalities=self.item_cat_cardinalities,
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


class GnnRanker(nn.Module):
    def __init__(self, cfg: GnnRankerConfig):
        super().__init__()
        self.cfg = cfg

        self.ctx_proj = (
            nn.Linear(cfg.num_ctx_feats, cfg.emb_dim, bias=False)
            if cfg.num_ctx_feats > 0
            else None
        )
        self.norm = nn.LayerNorm(cfg.emb_dim)

        self.gnn = GnnEncoder(cfg.gnn)
        self.ranker = Ranker(cfg.ranker)

    def forward(
        self,
        u_ids: torch.Tensor,
        h_ids: torch.Tensor,
        h_ctx: torch.Tensor,
        h_mask: torch.Tensor,
        c_ids: torch.Tensor,
        inter_graph: Data,
        u_static_feats: torch.Tensor,
        i_static_feats: torch.Tensor,
    ) -> torch.Tensor:
        user_embs, item_embs = self.gnn(inter_graph, u_static_feats, i_static_feats)

        padded_item_embs = torch.cat(
            [item_embs.new_zeros(1, item_embs.size(1)), item_embs], dim=0
        )
        padded_i_static = self._pad_item_static_features(i_static_feats)

        user_emb = user_embs[u_ids]
        candidate_emb = padded_item_embs[c_ids]
        hist_emb = self._build_hist_emb(padded_item_embs, h_ids, h_ctx, h_mask)

        u_feat_res = self.gnn.user_static_encoder(u_static_feats[u_ids])
        c_feat_res = self.gnn.item_static_encoder(padded_i_static[c_ids])

        user_emb = self.norm(user_emb + u_feat_res)
        candidate_emb = self.norm(candidate_emb + c_feat_res)

        scores = self.ranker(user_emb, hist_emb, candidate_emb, h_mask)
        return scores

    def _build_hist_emb(self, item_embs, history_ids, history_ctx, history_mask):
        hist_emb = item_embs[history_ids.clamp(min=0)]

        if self.ctx_proj is not None:
            ctx_emb = self.ctx_proj(history_ctx)
            hist_emb = self.norm(hist_emb + ctx_emb)

        return hist_emb * history_mask.unsqueeze(-1).float()

    def _pad_item_static_features(self, item_static: torch.Tensor) -> torch.Tensor:
        pad_row = item_static.new_zeros((1, item_static.size(1)))
        num_dense = self.cfg.num_item_dense_feats
        if len(self.cfg.item_cat_cardinalities) > 0:
            pad_row[:, num_dense:] = -1
        return torch.cat([pad_row, item_static], dim=0)
