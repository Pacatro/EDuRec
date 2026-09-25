import torch
from torch import nn

from ..configs import ModelConfig
from .base import BaseRecArch
from .modules.kg_encoder import GraphEncoder
from .modules.scorer import Scorer
from .modules.seq_encoder import SeqEncoder
from .modules.user_profile import UserProfileEncoder


class KGRNN(BaseRecArch):
    """Knowledge-graph educational recommender.

    The knowledge-graph encoder refines the item embeddings, each user's
    chronological history is gathered from those representations and encoded by
    a GRU or LSTM (``cfg.seq_encoder.cell_type``), and the resulting user state
    is scored against the item embeddings by a final MLP.

    When user features are available, a static profile is encoded separately and
    fused with the sequential state through a learned gate, so cold-start users
    can fall back on their profile.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.kg = GraphEncoder(cfg.kg_encoder)
        self.seq_encoder = SeqEncoder(cfg.seq_encoder)

        self.user_profile = (
            UserProfileEncoder(cfg.user_profile) if cfg.use_user_features else None
        )
        profile_active = self.user_profile is not None and cfg.user_profile.is_active
        self.fusion_gate = (
            nn.Linear(cfg.emb_dim * 2, cfg.emb_dim)
            if profile_active and cfg.user_fusion == "gate"
            else None
        )
        self.fusion_mlp = (
            nn.Sequential(
                nn.Linear(cfg.emb_dim * 2, cfg.emb_dim),
                nn.GELU(),
                nn.Linear(cfg.emb_dim, cfg.emb_dim),
            )
            if profile_active and cfg.user_fusion == "concat"
            else None
        )

        self.item_bias = (
            nn.Parameter(torch.zeros(cfg.num_items)) if cfg.use_item_bias else None
        )
        self.scorer = Scorer(cfg.scorer)

    def _fuse(self, user_emb: torch.Tensor, profile: torch.Tensor | None) -> torch.Tensor:
        if profile is None:
            return user_emb
        if self.fusion_gate is not None:
            joint = torch.cat([user_emb, profile], dim=-1)
            gate = torch.sigmoid(self.fusion_gate(joint))
            return gate * user_emb + (1.0 - gate) * profile
        if self.fusion_mlp is not None:
            return self.fusion_mlp(torch.cat([user_emb, profile], dim=-1))
        return user_emb

    def forward(
        self,
        h_ids: torch.Tensor,
        h_mask: torch.Tensor,
        edge_index: dict[tuple[str, str, str], torch.Tensor],
        i_static_feats: torch.Tensor,
        u_static_feats: torch.Tensor,
        u_cat_feats: torch.Tensor,
        user_ids: torch.Tensor,
        candidate_item_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        item_feats = i_static_feats[:, : self.cfg.effective_item_dense_feats]

        item_emb = self.kg(edge_index, item_feats)

        padded = torch.cat([item_emb.new_zeros(1, item_emb.size(1)), item_emb])
        hist = padded[h_ids.clamp(min=0)]

        profile = (
            self.user_profile(u_cat_feats, u_static_feats, user_ids)
            if self.user_profile is not None
            else None
        )
        user_emb = self.seq_encoder(hist, h_mask, condition=profile)
        user_emb = self._fuse(user_emb, profile)

        scores = self.scorer(user_emb, item_emb, item_ids=candidate_item_ids)

        if self.item_bias is not None:
            if candidate_item_ids is not None:
                scores = scores + self.item_bias[candidate_item_ids]
            else:
                scores = scores + self.item_bias

        return scores
