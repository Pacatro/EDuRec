from dataclasses import dataclass

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
    emb_dim: int
    num_user_feats: int
    num_item_feats: int
    num_ctx_feats: int

    # GCL Defaults
    edge_dropout: float = config.DROP_EDGES_P
    temperature: float = config.TAU
    max_user_samples: int = config.MAX_SAMPLES_U
    max_item_samples: int = config.MAX_SAMPLES_I
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
            embed_dim=self.emb_dim,
            drop_edges_p=self.edge_dropout,
            tau=self.temperature,
            max_samples_u=self.max_user_samples,
            max_samples_i=self.max_item_samples,
            loss_reduc=LossReduction(self.loss_reduction),
            num_layers=self.gnn_layers,
        )

    @property
    def ranker(self) -> RankerConfig:
        return RankerConfig(
            embed_dim=self.emb_dim,
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

        self.u_static_proj = nn.Linear(cfg.num_user_feats, cfg.emb_dim, bias=False)
        self.i_static_proj = nn.Linear(cfg.num_item_feats, cfg.emb_dim, bias=False)
        self.gnn = GnnEncoder(cfg.gnn)
        self.ranker = Ranker(cfg.ranker)

        self.register_buffer("edge_index", None)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.edge_index = inter_graph.edge_index

        user_embs, item_embs = self.gnn(inter_graph)

        user_emb = user_embs[u_ids]  # [B, D]
        hist_emb = self._build_hist_emb(item_embs, h_ids, h_ctx, h_mask)  # [B, H, D]
        candidate_emb = item_embs[c_ids]  # [B, C, D]

        # TODO: Dado que usamos ordinal encoder para las categóricas, lo ideal sería
        # separar las proyecciones de la siguiente manera:
        # - numericas -> linear
        # - categóricas -> embedding
        # y concatenar las dos proyecciones en un unico embedding final.
        user_feats = self.u_static_proj(u_static_global[u_ids])  # [B, D]
        hist_feats = self.i_static_proj(i_static_global[h_ids])  # [B, H, D]
        candidate_feats = self.i_static_proj(i_static_global[c_ids])  # [B, C, D]

        user_emb = user_emb + user_feats
        hist_emb = hist_emb + hist_feats
        candidate_emb = candidate_emb + candidate_feats

        scores = self.ranker(user_emb, hist_emb, candidate_emb)  # [B, C, num_scores]

        return scores, user_emb, item_embs

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
