import torch
from torch import nn

from ..configs import ModelConfig
from .fusion import FusionConfig, MaskedGatedFusion, SumFusion
from .kg_encoder import KGEncoder
from .mlp_encoder import MLPEncoder
from .scorer import Scorer
from .seq_encoder import SeqEncoder


class EDuRec(nn.Module):
    """Knowledge-graph educational recommender.

    User and item representations come from the knowledge-graph encoder and,
    for users, the sequential history encoder. Interaction context is encoded
    independently and consumed by the scorer.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        available = cfg.available_modules

        self.kg = KGEncoder(cfg.kg_encoder) if available["graph"] else None
        if self.kg is None:
            raise ValueError("The knowledge-graph encoder must always be active.")
        self.sequence_encoder = (
            SeqEncoder(cfg.seq_encoder) if available["sequence"] else None
        )
        self.context_encoder = (
            MLPEncoder(cfg.context_encoder) if available["context"] else None
        )

        user_sources = 1 + int(self.sequence_encoder is not None)
        self.user_fusion = self._make_fusion(user_sources)

        self.item_bias = (
            nn.Parameter(torch.zeros(cfg.num_items)) if cfg.use_item_bias else None
        )
        self.scorer = Scorer(cfg.scorer)

    def _make_fusion(self, num_sources: int) -> MaskedGatedFusion | SumFusion | None:
        # A single source needs neither gates nor normalization parameters.
        if num_sources == 1:
            return None
        fusion_cfg = FusionConfig(
            emb_dim=self.cfg.emb_dim,
            num_sources=num_sources,
            dropout=self.cfg.dropout,
        )
        if self.cfg.fusion_type == "sum":
            return SumFusion(fusion_cfg)
        if self.cfg.fusion_type == "masked_gated":
            return MaskedGatedFusion(fusion_cfg)
        raise ValueError(f"Unknown fusion type: {self.cfg.fusion_type!r}.")

    @staticmethod
    def _fuse(
        sources: list[torch.Tensor],
        fusion: MaskedGatedFusion | SumFusion | None,
        available: list[torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        """Fuse source representations, optionally masking unavailable ones."""
        if fusion is None:
            return sources[0]

        if available is not None and isinstance(fusion, MaskedGatedFusion):
            batch_size = sources[0].size(0)
            device = sources[0].device
            mask = torch.stack(
                [
                    torch.ones(batch_size, dtype=torch.bool, device=device)
                    if flag is None
                    else flag.bool()
                    for flag in available
                ],
                dim=1,
            )
            return fusion(sources, available=mask)

        return fusion(sources)

    def forward(
        self,
        u_ids: torch.Tensor,
        h_ids: torch.Tensor,
        h_mask: torch.Tensor,
        edge_index: dict[tuple[str, str, str], torch.Tensor],
        u_static_feats: torch.Tensor,
        i_static_feats: torch.Tensor,
        context: torch.Tensor | None = None,
        candidate_item_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        user_feats = u_static_feats[:, : self.cfg.effective_user_dense_feats]
        item_feats = i_static_feats[:, : self.cfg.effective_item_dense_feats]

        assert self.kg is not None
        user_graph, item_emb = self.kg(edge_index, user_feats, item_feats)

        user_sources = [user_graph[u_ids]]
        user_available: list[torch.Tensor | None] = [None]

        if self.sequence_encoder is not None:
            padded = torch.cat([item_emb.new_zeros(1, item_emb.size(1)), item_emb])
            hist = padded[h_ids.clamp(min=0)]
            seq_user = self.sequence_encoder(hist, h_mask)
            user_sources.append(seq_user)
            user_available.append(h_mask.bool().any(dim=1))

        context_emb = None
        if self.context_encoder is not None:
            if context is None:
                raise ValueError(
                    "context is required when the context module is active."
                )
            context_emb = self.context_encoder(context)

        user_emb = self._fuse(user_sources, self.user_fusion, user_available)

        scores = self.scorer(
            user_emb,
            item_emb,
            context_emb,
            item_ids=candidate_item_ids,
        )

        if self.item_bias is not None:
            if candidate_item_ids is not None:
                scores = scores + self.item_bias[candidate_item_ids]
            else:
                scores = scores + self.item_bias

        return scores
