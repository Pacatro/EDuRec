import torch
from torch import nn

from ..configs import ModelConfig
from .base import BaseRecArch
from .modules.kg_encoder import KGEncoder
from .modules.scorer import Scorer
from .modules.seq_encoder import SeqEncoder


class KGSeq(BaseRecArch):
    """Serial knowledge-graph to sequence recommender.

    The knowledge-graph encoder processes the full graph and produces item node
    representations. For each user, the item nodes that appear in their
    chronological history are gathered from those graph representations and
    encoded by the sequential encoder. Its output is the only user
    representation fed to the scorer, which scores it against the item
    embeddings.

    Unlike ``KGRNN``, the graph user node is not concatenated with the sequence
    state: the graph feeds the sequence, not the scorer.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.kg = KGEncoder(cfg.kg_encoder)
        self.sequence_encoder = SeqEncoder(cfg.seq_encoder)

        self.item_bias = (
            nn.Parameter(torch.zeros(cfg.num_items)) if cfg.use_item_bias else None
        )
        self.scorer = Scorer(cfg.scorer)

    def forward(
        self,
        u_ids: torch.Tensor,
        h_ids: torch.Tensor,
        h_mask: torch.Tensor,
        edge_index: dict[tuple[str, str, str], torch.Tensor],
        u_static_feats: torch.Tensor,
        i_static_feats: torch.Tensor,
        candidate_item_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        user_feats = u_static_feats[:, : self.cfg.effective_user_dense_feats]
        item_feats = i_static_feats[:, : self.cfg.effective_item_dense_feats]

        user_graph, item_emb = self.kg(edge_index, user_feats, item_feats)
        user_emb = self._user_representation(user_graph, item_emb, u_ids, h_ids, h_mask)

        scores = self.scorer(user_emb, item_emb, item_ids=candidate_item_ids)

        if self.item_bias is not None:
            if candidate_item_ids is not None:
                scores = scores + self.item_bias[candidate_item_ids]
            else:
                scores = scores + self.item_bias

        return scores

    def _user_representation(
        self,
        user_graph: torch.Tensor,
        item_emb: torch.Tensor,
        u_ids: torch.Tensor,
        h_ids: torch.Tensor,
        h_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the graph-derived item history of each user.

        The serial architecture is only built when history is available
        (``build_model`` enforces it); the graph user node is kept as a safe
        fallback so the module never dereferences missing history tensors.
        """
        if not self.cfg.uses_sequence:
            return user_graph[u_ids]

        padded = torch.cat([item_emb.new_zeros(1, item_emb.size(1)), item_emb])
        hist = padded[h_ids.clamp(min=0)]
        return self.sequence_encoder(hist, h_mask)
