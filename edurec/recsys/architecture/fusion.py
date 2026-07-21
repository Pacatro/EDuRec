from dataclasses import dataclass

import torch
from torch import nn

from ... import settings


@dataclass
class FusionConfig:
    emb_dim: int = settings.EMB_DIM
    num_sources: int = 3
    dropout: float = settings.DROPOUT


class MaskedGatedFusion(nn.Module):
    def __init__(self, cfg: FusionConfig) -> None:
        super().__init__()

        self.cfg = cfg

        self.source_norms = nn.ModuleList(
            nn.LayerNorm(cfg.emb_dim) for _ in range(cfg.num_sources)
        )

        self.gate_logits = nn.Parameter(torch.zeros(cfg.num_sources))

        self.dropout = nn.Dropout(cfg.dropout)
        self.output_norm = nn.LayerNorm(cfg.emb_dim)

    def forward(
        self,
        sources: list[torch.Tensor],
        available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = sources[0].size(0)
        device = sources[0].device

        if available is None:
            available = torch.ones(
                batch_size,
                self.cfg.num_sources,
                dtype=torch.bool,
                device=device,
            )
        else:
            expected_shape = (batch_size, self.cfg.num_sources)
            if available.shape != expected_shape:
                raise ValueError(
                    f"available must have shape {expected_shape}, "
                    f"got {tuple(available.shape)}."
                )
            available = available.to(device=device, dtype=torch.bool)

        normalized_sources = torch.stack(
            [norm(source) for norm, source in zip(self.source_norms, sources)],
            dim=1,
        )

        gate_logits = self.gate_logits.expand(batch_size, -1).masked_fill(
            ~available,
            torch.finfo(self.gate_logits.dtype).min,
        )
        has_available_source = available.any(dim=1)
        weights = torch.softmax(gate_logits, dim=1).masked_fill(~available, 0.0)

        fused = (normalized_sources * weights.unsqueeze(-1)).sum(dim=1)
        fused = self.output_norm(self.dropout(fused))
        return fused.masked_fill(~has_available_source.unsqueeze(-1), 0.0)
