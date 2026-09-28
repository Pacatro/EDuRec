import math

import torch
from torch import nn

from ..configs import ModelConfig
from .base import BaseRecArch
from .modules.candidate_attention import CandidateAttention
from .modules.gated_fusion import GatedFusion
from .modules.history_transformer import HistoryTransformer
from .modules.kg_encoder import GraphEncoder
from .modules.reranker import CandidateReranker
from .modules.user_profile import UserProfileEncoder


class GraphTransformer(BaseRecArch):
    """Two-stage educational recommender over a knowledge graph.

    The relation-aware GraphSAGE encoder (``self.kg``) refines item embeddings
    from the course identifier, metadata and multilingual text features. A
    Transformer encodes each user's chronological history into a global user
    representation (used for the dot-product retrieval) and a memory sequence.

    Retrieval keeps the ``cfg.retrieval_k`` best items by dot product. Those
    candidates are then attended over the history memory, fused with the static
    user profile through a learned gate and finally scored by an MLP reranker.
    Candidates are scored directly during training, where the candidate set
    already plays the role of the retrieval shortlist.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.kg = GraphEncoder(cfg.kg_encoder)
        self.history_transformer = HistoryTransformer(cfg.history_transformer)
        self.candidate_attention = CandidateAttention(cfg.candidate_attention)
        self.reranker = CandidateReranker(cfg.reranker)

        self.user_profile = (
            UserProfileEncoder(cfg.user_profile) if cfg.use_user_features else None
        )
        profile_active = self.user_profile is not None and cfg.user_profile.is_active
        self.profile_fusion = GatedFusion(cfg.emb_dim) if profile_active else None

        self.item_bias = (
            nn.Parameter(torch.zeros(cfg.num_items)) if cfg.use_item_bias else None
        )
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(cfg.emb_dim)))

    def _retrieval_scores(
        self, user_emb: torch.Tensor, item_emb: torch.Tensor
    ) -> torch.Tensor:
        return user_emb @ item_emb.T * self.logit_scale.clamp(min=1e-3)

    def _candidate_retrieval(
        self, user_emb: torch.Tensor, candidate_emb: torch.Tensor
    ) -> torch.Tensor:
        scores = torch.bmm(user_emb.unsqueeze(1), candidate_emb.transpose(1, 2))
        return scores.squeeze(1) * self.logit_scale.clamp(min=1e-3)

    def _rerank_scores(
        self,
        candidate_emb: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        user_emb: torch.Tensor,
    ) -> torch.Tensor:
        context = self.candidate_attention(candidate_emb, memory, memory_mask)
        return self.reranker(candidate_emb, context, user_emb)

    def _score_candidates(
        self,
        candidate_item_ids: torch.Tensor,
        item_emb: torch.Tensor,
        global_user: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        rerank_user: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_candidates = candidate_item_ids.shape
        if num_candidates == 0:
            return memory.new_empty((batch_size, 0))

        candidate_emb = item_emb[candidate_item_ids]
        rerank = self._rerank_scores(candidate_emb, memory, memory_mask, rerank_user)
        retrieval = self._candidate_retrieval(global_user, candidate_emb)

        # The retrieval term is kept as a residual so the dot-product tower is
        # trained even when only the shortlist is scored.
        scores = rerank + retrieval
        if self.item_bias is not None:
            scores = scores + self.item_bias[candidate_item_ids]
        return scores

    def _retrieve_and_rerank(
        self,
        item_emb: torch.Tensor,
        retrieval: torch.Tensor,
        global_user: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        rerank_user: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_items = retrieval.shape
        top_k = min(max(self.cfg.retrieval_k, 1), num_items)

        _, top_ids = torch.topk(retrieval, k=top_k, dim=1, largest=True, sorted=True)
        top_emb = item_emb[top_ids]

        top_scores = self._rerank_scores(top_emb, memory, memory_mask, rerank_user)
        top_scores = top_scores + torch.gather(retrieval, 1, top_ids)
        if self.item_bias is not None:
            top_scores = top_scores + self.item_bias[top_ids]

        # Non-retrieved items are excluded from the ranking. ``finfo.min`` keeps
        # the score matrix finite for the metric implementation.
        scores = retrieval.new_full(
            (batch_size, num_items), torch.finfo(retrieval.dtype).min
        )
        scores.scatter_(1, top_ids, top_scores)
        return scores

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
        history = padded[h_ids.clamp(min=0)]

        profile = (
            self.user_profile(u_cat_feats, u_static_feats, user_ids)
            if self.user_profile is not None
            else None
        )
        global_user, memory = self.history_transformer(history, h_mask)

        rerank_user = (
            self.profile_fusion(global_user, profile)
            if self.profile_fusion is not None and profile is not None
            else global_user
        )

        if candidate_item_ids is not None:
            return self._score_candidates(
                candidate_item_ids.long(),
                item_emb,
                global_user,
                memory,
                h_mask,
                rerank_user,
            )

        retrieval = self._retrieval_scores(global_user, item_emb)
        return self._retrieve_and_rerank(
            item_emb, retrieval, global_user, memory, h_mask, rerank_user
        )
