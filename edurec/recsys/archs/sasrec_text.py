import torch
from torch import nn

from edurec.recsys.archs.base import BaseRecArch
from edurec.recsys.archs.modules import TransformerSeqEncoder
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
        self.seq_encoder = TransformerSeqEncoder(
            emb_dim=cfg.emb_dim,
            hidden_dim=cfg.transformer_hidden_dim,
            num_layers=cfg.transformer_layers,
            num_heads=cfg.transformer_heads,
            dropout=cfg.dropout,
            max_history_len=cfg.max_history_len,
            causal=True,
            use_last_state=True,
        )

    def _compute_item_embeddings(self) -> torch.Tensor:
        embeddings = self.item_embedding.weight
        if self.content_proj is not None:
            embeddings = embeddings + self.content_proj(self.item_dense_feats)
        return embeddings

    def forward(
        self,
        h_ids: torch.Tensor,
        h_mask: torch.Tensor,
        user_ids: torch.Tensor,
        h_dense: torch.Tensor | None = None,
        h_cat: torch.Tensor | None = None,
        h_delta: torch.Tensor | None = None,
        candidate_item_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # SASRecText is content+ID only; interaction context and time are unused.
        course_emb = self.item_embeddings()
        padded = torch.cat([course_emb.new_zeros(1, course_emb.size(1)), course_emb])
        user_state = self.seq_encoder(padded[h_ids.clamp(min=0)], h_mask)
        if candidate_item_ids is None:
            return user_state @ course_emb.T
        candidates = course_emb[candidate_item_ids]
        if candidate_item_ids.ndim == 1:
            return user_state @ candidates.T
        return (user_state.unsqueeze(1) * candidates).sum(dim=-1)
