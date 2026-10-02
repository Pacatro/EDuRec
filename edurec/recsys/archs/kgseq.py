import torch
from torch import nn

from edurec.recsys.archs.base import BaseRecArch
from edurec.recsys.archs.modules.graph_user_fusion import GraphUserFusion
from edurec.recsys.archs.modules.kg_encoder import GraphEncoder
from edurec.recsys.archs.modules.scorer import Scorer
from edurec.recsys.archs.modules.seq_encoder import SeqEncoder
from edurec.recsys.archs.modules.user_profile import UserProfileEncoder
from edurec.recsys.configs import ModelConfig


class KGSeq(BaseRecArch):
    """Knowledge-graph educational recommender.

    The collaborative knowledge-graph encoder refines users and items. Each user's
    chronological history is gathered from those representations and encoded by
    a GRU or LSTM (``cfg.seq_encoder.cell_type``), and the resulting user state
    is fused with the graph user before profile fusion and item scoring.

    When user features are available, a static profile is encoded separately and
    fused with the sequential state through a learned gate, so cold-start users
    can fall back on their profile.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.kg = GraphEncoder(cfg.kg_encoder)
        self.graph_user_fusion = GraphUserFusion(cfg.emb_dim)
        self.seq_encoder = self._build_seq_encoder(cfg)

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

    def _build_seq_encoder(self, cfg: ModelConfig) -> nn.Module:
        """Build the sequential encoder; subclasses may swap in another module."""
        return SeqEncoder(cfg.seq_encoder)

    def _fuse(
        self, user_emb: torch.Tensor, profile: torch.Tensor | None
    ) -> torch.Tensor:
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

        item_emb, graph_users = self.kg(edge_index, item_feats)

        padded = torch.cat([item_emb.new_zeros(1, item_emb.size(1)), item_emb])
        hist = padded[h_ids.clamp(min=0)]

        profile = (
            self.user_profile(u_cat_feats, u_static_feats, user_ids)
            if self.user_profile is not None
            else None
        )
        history_states = None
        if self.cfg.scorer_type == "candidate_attention":
            user_emb, history_states = self.seq_encoder(
                hist, h_mask, condition=profile, return_sequence=True
            )
        else:
            user_emb = self.seq_encoder(hist, h_mask, condition=profile)
        has_history = h_mask.bool().any(dim=1)
        user_emb, has_graph = self.graph_user_fusion(
            user_emb,
            graph_users,
            user_ids,
            has_history,
            edge_index,
            enabled=self.cfg.graph_mode == "kg",
        )
        user_emb = self._fuse(user_emb, profile)
        if profile is not None:
            user_emb = torch.where(
                (has_history | has_graph)[:, None], user_emb, profile
            )

        scores = self.scorer(
            user_emb,
            item_emb,
            item_ids=candidate_item_ids,
            history_states=history_states,
            history_mask=h_mask if history_states is not None else None,
        )

        if self.item_bias is not None:
            if candidate_item_ids is not None:
                scores = scores + self.item_bias[candidate_item_ids]
            else:
                scores = scores + self.item_bias

        return scores
