import torch
from torch import nn

from ..configs import ModelConfig
from .fusion import FusionConfig, MaskedGatedFusion, SumFusion
from .graph_encoder import GraphEncoder
from .mlp_encoder import MLPEncoder
from .scorer import Scorer
from .seq_encoder import SeqEncoder


class EDuRec(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        available = cfg.available_modules

        self.gnn = GraphEncoder(cfg.graph_encoder) if available["graph"] else None
        self.user_encoder = (
            MLPEncoder(cfg.user_encoder) if available["user_features"] else None
        )
        self.item_encoder = (
            MLPEncoder(cfg.item_encoder) if available["item_features"] else None
        )
        self.context_encoder = (
            MLPEncoder(cfg.context_encoder) if available["context"] else None
        )
        item_sources = int(available["graph"]) + int(available["item_features"])
        user_sources = (
            int(available["graph"])
            + int(available["user_features"])
            + int(available["sequence"])
        )
        if item_sources == 0 or user_sources == 0:
            raise ValueError(
                "The effective configuration must provide at least one user and "
                "one item representation module."
            )
        self.item_fusion = self._make_fusion(item_sources)
        self.user_fusion = self._make_fusion(user_sources)
        self.item_bias = (
            nn.Parameter(torch.zeros(cfg.num_items)) if cfg.use_item_bias else None
        )
        self.sequence_encoder = (
            SeqEncoder(cfg.seq_encoder) if available["sequence"] else None
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
    ) -> torch.Tensor:
        return sources[0] if fusion is None else fusion(sources)

    def forward(
        self,
        u_ids: torch.Tensor,
        h_ids: torch.Tensor,
        h_mask: torch.Tensor,
        edge_index: torch.Tensor,
        u_static_feats: torch.Tensor,
        i_static_feats: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        user_sources = []
        item_sources = []

        if self.gnn is not None:
            user_graph, item_graph = self.gnn(edge_index)
            user_sources.append(user_graph[u_ids])
            item_sources.append(item_graph)

        if self.user_encoder is not None:
            user_sources.append(self.user_encoder(u_static_feats)[u_ids])
        if self.item_encoder is not None:
            item_sources.append(self.item_encoder(i_static_feats))
        context_emb = None
        if self.context_encoder is not None:
            if context is None:
                raise ValueError(
                    "context is required when the context module is active."
                )
            context_emb = self.context_encoder(context)

        item_emb = self._fuse(item_sources, self.item_fusion)

        if self.sequence_encoder is not None:
            padded = torch.cat([item_emb.new_zeros(1, item_emb.size(1)), item_emb])
            hist = padded[h_ids.clamp(min=0)]
            seq_user = self.sequence_encoder(hist, h_mask)
            user_sources.append(seq_user)

        user_emb = self._fuse(user_sources, self.user_fusion)

        scores = self.scorer(user_emb, item_emb, context_emb)

        if self.item_bias is not None:
            scores = scores + self.item_bias

        return scores
