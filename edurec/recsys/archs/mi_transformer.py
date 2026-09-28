import torch
from torch import nn

from ..configs import ModelConfig
from .base import BaseRecArch
from .modules.causal_transformer import CausalTransformer
from .modules.item_encoder import ItemEncoder
from .modules.multi_interest import (
    InterestProfileGate,
    MultiInterestAttention,
    MultiInterestScorer,
)
from .modules.user_profile import UserProfileEncoder


class MITransformer(BaseRecArch):
    """Multi-interest educational recommender.

    Each item is encoded from its course identifier, numeric metadata and
    multilingual text embedding, then refined by relation-aware GraphSAGE over
    the schema knowledge graph. A user's chronological history is encoded by a
    causal transformer with positional and temporal information, and a bank of
    learned queries pools the hidden states into ``K`` interest vectors. An
    optional static profile is gated into those interests, and every item is
    scored by its best-matching interest (multi-interest dot product).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.item_encoder = ItemEncoder(cfg.item_encoder)
        self.seq_encoder = CausalTransformer(cfg.transformer)
        self.multi_interest = MultiInterestAttention(cfg.multi_interest)

        self.user_profile = (
            UserProfileEncoder(cfg.user_profile) if cfg.use_user_features else None
        )
        profile_active = self.user_profile is not None and cfg.user_profile.is_active
        self.profile_gate = InterestProfileGate(cfg.emb_dim) if profile_active else None

        self.item_bias = (
            nn.Parameter(torch.zeros(cfg.num_items)) if cfg.use_item_bias else None
        )
        self.scorer = MultiInterestScorer(cfg.multi_interest)

    def forward(
        self,
        h_ids: torch.Tensor,
        h_mask: torch.Tensor,
        h_times: torch.Tensor,
        edge_index: dict[tuple[str, str, str], torch.Tensor],
        i_static_feats: torch.Tensor,
        u_static_feats: torch.Tensor,
        u_cat_feats: torch.Tensor,
        user_ids: torch.Tensor,
        candidate_item_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        item_feats = i_static_feats[:, : self.cfg.effective_item_dense_feats]
        item_emb = self.item_encoder(edge_index, item_feats)

        padded = torch.cat([item_emb.new_zeros(1, item_emb.size(1)), item_emb])
        history = padded[h_ids.clamp(min=0)]

        hidden = self.seq_encoder(history, h_mask, history_times=h_times)
        interests = self.multi_interest(hidden, h_mask)

        profile = (
            self.user_profile(u_cat_feats, u_static_feats, user_ids)
            if self.user_profile is not None
            else None
        )
        if profile is not None and self.profile_gate is not None:
            interests = self.profile_gate(interests, profile)

        scores = self.scorer(interests, item_emb, item_ids=candidate_item_ids)

        if self.item_bias is not None:
            if candidate_item_ids is not None:
                scores = scores + self.item_bias[candidate_item_ids]
            else:
                scores = scores + self.item_bias

        return scores
