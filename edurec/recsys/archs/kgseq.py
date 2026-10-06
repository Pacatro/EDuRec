import torch
from torch import nn

from edurec.recsys.archs.base import BaseRecArch
from edurec.recsys.archs.modules.interaction_context import InteractionContextEncoder
from edurec.recsys.archs.modules.kg_encoder import GraphEncoder
from edurec.recsys.archs.modules.scorer import Scorer
from edurec.recsys.archs.modules.seq_encoder import SeqEncoder
from edurec.recsys.archs.modules.user_state import UserStateEncoder
from edurec.recsys.configs import ModelConfig


class TimeEncoder(nn.Module):
    """Project non-negative time gaps after logarithmic compression."""

    def __init__(self, emb_dim: int):
        super().__init__()
        self.proj = nn.Linear(1, emb_dim)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.log1p(delta.clamp(min=0.0)).unsqueeze(-1))


class KGSeq(BaseRecArch):
    """Item-knowledge-graph educational recommender with a recurrent encoder.

    Item relations are encoded by a relation-aware item GNN that never sees
    user-item interactions. Each historical event is the sum of its item
    embedding, its interaction context and its time gap:

    ``x_t = LayerNorm(item_emb[t] + context[t] + time[t])``

    The static user profile initialises the recurrent state
    (``h0 = P_user(profile)``), the GRU/LSTM produces the user state (last valid
    hidden state by default), and an MLP scores user/candidate pairs.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.kg = GraphEncoder(cfg.kg_encoder)
        self.interaction_context = InteractionContextEncoder(cfg.interaction_context)
        self.time_encoder = TimeEncoder(cfg.emb_dim) if cfg.use_time_features else None
        self.user_state = UserStateEncoder(cfg.user_state)
        self.event_norm = nn.LayerNorm(cfg.emb_dim)

        self.seq_encoder = SeqEncoder(cfg.seq_encoder)

        self.item_bias = (
            nn.Parameter(torch.zeros(cfg.num_items)) if cfg.use_item_bias else None
        )
        self.scorer = Scorer(cfg.scorer)

    def forward(
        self,
        h_ids: torch.Tensor,
        h_mask: torch.Tensor,
        edge_index: dict[tuple[str, str, str], torch.Tensor],
        i_static_feats: torch.Tensor,
        u_static_feats: torch.Tensor,
        u_cat_feats: torch.Tensor,
        user_ids: torch.Tensor,
        h_dense: torch.Tensor | None = None,
        h_cat: torch.Tensor | None = None,
        h_delta: torch.Tensor | None = None,
        candidate_item_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        item_feats = i_static_feats[:, : self.cfg.effective_item_dense_feats]
        item_emb = self.kg(edge_index, item_feats)

        padded = torch.cat([item_emb.new_zeros(1, item_emb.size(1)), item_emb])
        event = padded[h_ids.clamp(min=0)]

        if h_dense is not None and h_cat is not None:
            context = self.interaction_context(h_dense, h_cat)
            if context is not None:
                event = event + context

        if self.time_encoder is not None and h_delta is not None:
            event = event + self.time_encoder(h_delta)

        event = self.event_norm(event)

        initial_state = self.user_state(u_cat_feats, u_static_feats, user_ids)
        user_emb = self.seq_encoder(event, h_mask, initial_state=initial_state)

        scores = self.scorer(user_emb, item_emb, item_ids=candidate_item_ids)

        if self.item_bias is not None:
            scores += (
                self.item_bias[candidate_item_ids]
                if candidate_item_ids is not None
                else self.item_bias
            )

        return scores
