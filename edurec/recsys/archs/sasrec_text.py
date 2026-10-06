import torch
from torch import nn

from edurec.recsys.archs.base import BaseRecArch
from edurec.recsys.archs.modules.seq_encoder import TransformerSeqEncoder
from edurec.recsys.configs import ModelConfig


class SASRecText(BaseRecArch):
    """ID + pretrained content embeddings, causal history, and dot scoring.

    Content vectors are supplied by the existing preprocessing pipeline:
    numeric metadata followed by frozen pretrained text embeddings. A learned
    projection aligns them with course IDs. History IDs are one-based (zero is
    padding); candidate IDs use the zero-based catalog indices.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.item_embedding = nn.Embedding(cfg.num_items, cfg.emb_dim)
        self.content_proj = (
            nn.Linear(cfg.effective_item_dense_feats, cfg.emb_dim, bias=False)
            if cfg.effective_item_dense_feats > 0
            else None
        )
        self.seq_encoder = TransformerSeqEncoder(cfg.transformer_seq_encoder)

    def course_embeddings(self, i_static_feats: torch.Tensor) -> torch.Tensor:
        """Return the shared [num_items, emb_dim] history/candidate table."""
        embeddings = self.item_embedding.weight
        if self.content_proj is not None:
            content = i_static_feats[:, : self.cfg.effective_item_dense_feats]
            embeddings = embeddings + self.content_proj(content)
        return embeddings

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
        course_emb = self.course_embeddings(i_static_feats)
        padded = torch.cat([course_emb.new_zeros(1, course_emb.size(1)), course_emb])
        user_state = self.seq_encoder(padded[h_ids.clamp(min=0)], h_mask)
        if candidate_item_ids is None:
            return user_state @ course_emb.T
        candidates = course_emb[candidate_item_ids]
        if candidate_item_ids.ndim == 1:
            return user_state @ candidates.T
        return (user_state.unsqueeze(1) * candidates).sum(dim=-1)
