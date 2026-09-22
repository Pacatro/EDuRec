import torch
from torch import nn

from ..configs import ModelConfig
from .modules.kg_encoder import KGEncoder
from .modules.scorer import Scorer
from .modules.seq_encoder import SeqEncoder


class KGRNN(nn.Module):
    """Knowledge-graph educational recommender.

    User representations come from the knowledge-graph encoder and, when
    history is available, from the sequential history encoder. Both
    representations are concatenated and scored against the item embeddings by
    a final MLP.

    The sequential encoder uses a GRU or LSTM depending on
    ``cfg.seq_encoder.cell_type`` (``seq_cell`` in ``ModelConfig``).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        available = cfg.available_modules

        self.kg = KGEncoder(cfg.kg_encoder)
        self.sequence_encoder = (
            SeqEncoder(cfg.seq_encoder) if available["sequence"] else None
        )

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
        user_emb = user_graph[u_ids]

        if self.sequence_encoder is not None:
            padded = torch.cat([item_emb.new_zeros(1, item_emb.size(1)), item_emb])
            hist = padded[h_ids.clamp(min=0)]
            seq_user = self.sequence_encoder(hist, h_mask)
            user_emb = torch.cat([user_emb, seq_user], dim=-1)

        scores = self.scorer(user_emb, item_emb, item_ids=candidate_item_ids)

        if self.item_bias is not None:
            if candidate_item_ids is not None:
                scores = scores + self.item_bias[candidate_item_ids]
            else:
                scores = scores + self.item_bias

        return scores
