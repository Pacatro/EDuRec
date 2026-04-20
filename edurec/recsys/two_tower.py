from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .. import settings
from .static_feats_encoder import StaticFeatureEncoder


@dataclass
class RetrievalConfig:
    num_users: int
    num_items: int
    num_ctx_feats: int

    emb_dim: int = settings.EMB_DIM
    proj_dim: int = settings.EMB_DIM
    dropout: float = settings.DROPOUT
    temperature: float = settings.TAU

    num_user_dense_feats: int = 0
    num_item_dense_feats: int = 0
    user_cat_cardinalities: list[int] = field(default_factory=list)
    item_cat_cardinalities: list[int] = field(default_factory=list)

    use_history_ctx: bool = True


class MLPProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        hidden_dim = max(in_dim, out_dim)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoTowerRetrieval(nn.Module):
    def __init__(self, cfg: RetrievalConfig):
        super().__init__()
        self.cfg = cfg

        self.user_id_embedding = nn.Embedding(cfg.num_users, cfg.emb_dim)

        self.item_id_embedding = nn.Embedding(
            cfg.num_items + 1, cfg.emb_dim, padding_idx=0
        )

        self.user_static_encoder = StaticFeatureEncoder(
            num_dense_features=cfg.num_user_dense_feats,
            categorical_cardinalities=cfg.user_cat_cardinalities,
            emb_dim=cfg.emb_dim,
        )
        self.item_static_encoder = StaticFeatureEncoder(
            num_dense_features=cfg.num_item_dense_feats,
            categorical_cardinalities=cfg.item_cat_cardinalities,
            emb_dim=cfg.emb_dim,
        )

        self.ctx_proj = (
            nn.Linear(cfg.num_ctx_feats, cfg.emb_dim, bias=False)
            if cfg.num_ctx_feats > 0 and cfg.use_history_ctx
            else None
        )

        num_query_blocks = 4 + (1 if self.ctx_proj is not None else 0)

        self.query_proj = MLPProjection(
            in_dim=num_query_blocks * cfg.emb_dim,
            out_dim=cfg.proj_dim,
            dropout=cfg.dropout,
        )

        self.item_proj = MLPProjection(
            in_dim=2 * cfg.emb_dim,
            out_dim=cfg.proj_dim,
            dropout=cfg.dropout,
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        history_items: torch.Tensor,
        history_ctx: torch.Tensor,
        history_valid_mask: torch.Tensor,
        item_ids: torch.Tensor,
        u_static_feats: torch.Tensor,
        i_static_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_emb = self.encode_query(
            user_ids=user_ids,
            history_items=history_items,
            history_ctx=history_ctx,
            history_valid_mask=history_valid_mask,
            u_static_feats=u_static_feats,
            i_static_feats=i_static_feats,
        )
        item_emb = self.encode_items(
            item_ids=item_ids,
            i_static_feats=i_static_feats,
        )
        return query_emb, item_emb

    def encode_query(
        self,
        user_ids: torch.Tensor,
        history_items: torch.Tensor,
        history_ctx: torch.Tensor,
        history_valid_mask: torch.Tensor,
        u_static_feats: torch.Tensor,
        i_static_feats: torch.Tensor,
    ) -> torch.Tensor:
        user_static_repr = self.user_static_encoder(u_static_feats[user_ids])
        user_id_repr = self.user_id_embedding(user_ids)

        padded_item_static = self._pad_item_static_features(i_static_feats)
        history_static_repr = self.item_static_encoder(
            padded_item_static[history_items]
        )

        history_item_repr = self.item_id_embedding(history_items.clamp(min=0))

        history_mask = history_valid_mask.unsqueeze(-1).float()

        pieces = [
            user_id_repr,
            user_static_repr,
            self._masked_mean(history_item_repr, history_mask),
            self._masked_mean(history_static_repr, history_mask),
        ]

        if self.ctx_proj is not None:
            pieces.append(
                self._masked_mean(self.ctx_proj(history_ctx.float()), history_mask)
            )

        query_emb = self.query_proj(torch.cat(pieces, dim=-1))
        return F.normalize(query_emb, dim=-1)

    def encode_items(
        self,
        item_ids: torch.Tensor,
        i_static_feats: torch.Tensor,
    ) -> torch.Tensor:
        item_ids = item_ids.long()

        if item_ids.ndim not in (1, 2):
            raise RuntimeError("item_ids must be a 1D or 2D tensor.")

        flat_ids = item_ids.reshape(-1)
        item_static_repr = self.item_static_encoder(i_static_feats[flat_ids])
        item_id_repr = self.item_id_embedding(flat_ids + 1)

        item_emb = self.item_proj(torch.cat([item_id_repr, item_static_repr], dim=-1))
        item_emb = F.normalize(item_emb, dim=-1)

        if item_ids.ndim == 1:
            return item_emb

        return item_emb.view(*item_ids.shape, -1)

    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (x * mask).sum(dim=1) / denom

    def _pad_item_static_features(self, item_static: torch.Tensor) -> torch.Tensor:
        if item_static.ndim != 2:
            raise RuntimeError("Item static features must be a 2D tensor.")

        pad_row = item_static.new_zeros((1, item_static.size(1)))

        num_dense = self.cfg.num_item_dense_feats
        num_cats = len(self.cfg.item_cat_cardinalities)

        if num_cats > 0:
            # categorical unknown/padding -> -1, luego +1 en encoder => 0
            pad_row[:, num_dense : num_dense + num_cats] = -1

        return torch.cat([pad_row, item_static], dim=0)
