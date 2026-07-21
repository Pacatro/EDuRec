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
        self._validate_sources(sources)

        batch_size = sources[0].size(0)

        if available is None:
            available = torch.ones(
                batch_size,
                self.cfg.num_sources,
                dtype=torch.bool,
                device=sources[0].device,
            )
        else:
            expected_shape = (batch_size, self.cfg.num_sources)

            if available.shape != expected_shape:
                raise ValueError(
                    f"available must have shape {expected_shape}, "
                    f"got {tuple(available.shape)}."
                )

            available = available.to(device=sources[0].device, dtype=torch.bool)

        normalized_sources = torch.stack(
            [self.source_norms[i](sources[i]) for i in range(self.cfg.num_sources)],
            dim=1,
        )

        gate_logits = self.gate_logits.unsqueeze(0).expand(batch_size, -1)
        gate_logits = gate_logits.masked_fill(
            ~available,
            torch.finfo(gate_logits.dtype).min,
        )
        has_available_source = available.any(dim=1)
        weights = torch.softmax(gate_logits, dim=1)
        weights = weights.masked_fill(~has_available_source.unsqueeze(1), 0.0)

        fused = torch.sum(
            normalized_sources * weights.unsqueeze(-1),
            dim=1,
        )

        fused = self.output_norm(self.dropout(fused))

        fused = fused.masked_fill(
            ~has_available_source.unsqueeze(-1),
            0.0,
        )

        return fused

    def _validate_sources(self, sources: list[torch.Tensor]) -> None:
        if len(sources) != self.cfg.num_sources:
            raise ValueError(
                f"Expected {self.cfg.num_sources} sources, got {len(sources)}."
            )

        if not sources:
            raise ValueError("At least one source is required.")

        batch_size = sources[0].size(0)
        expected_shape = (batch_size, self.cfg.emb_dim)

        for i, source in enumerate(sources):
            if source.shape != expected_shape:
                raise ValueError(
                    f"Source {i} must have shape {expected_shape}, "
                    f"got {tuple(source.shape)}."
                )
