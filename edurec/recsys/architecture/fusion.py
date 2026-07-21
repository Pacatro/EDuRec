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

        self.gates = nn.ModuleList(
            nn.Linear(cfg.emb_dim, 1) for _ in range(cfg.num_sources)
        )

        self.dropout = nn.Dropout(cfg.dropout)
        self.output_norm = nn.LayerNorm(cfg.emb_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for gate in self.gates:
            assert isinstance(gate, nn.Linear)
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)

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

            available = available.bool()

        normalized_sources = torch.stack(
            [self.source_norms[i](sources[i]) for i in range(self.cfg.num_sources)],
            dim=1,
        )

        gate_logits = torch.cat(
            [
                self.gates[i](normalized_sources[:, i])
                for i in range(self.cfg.num_sources)
            ],
            dim=1,
        )

        gates = torch.sigmoid(gate_logits)
        gates = gates * available.to(gates.dtype)

        has_available_source = available.any(dim=1)

        weights = gates / gates.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-8)

        weights = weights.masked_fill(
            ~has_available_source.unsqueeze(1),
            0.0,
        )

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
