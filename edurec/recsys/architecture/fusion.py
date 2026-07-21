from dataclasses import dataclass

import torch
from torch import nn

from ... import settings


@dataclass
class FusionConfig:
    emb_dim: int = settings.EMB_DIM
    num_sources: int = 3
    dropout: float = settings.DROPOUT
    n_heads: int = 4


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


class SumFusion(nn.Module):
    def __init__(self, cfg: FusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.source_norms = nn.ModuleList(
            nn.LayerNorm(cfg.emb_dim) for _ in range(cfg.num_sources)
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.output_norm = nn.LayerNorm(cfg.emb_dim)

    def forward(
        self,
        sources: list[torch.Tensor],
        available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_sources(sources)

        if available is None:
            normalized = [
                self.source_norms[i](sources[i]) for i in range(self.cfg.num_sources)
            ]
            fused = sum(normalized)
        else:
            available = available.bool()
            normalized = torch.stack(
                [self.source_norms[i](sources[i]) for i in range(self.cfg.num_sources)],
                dim=1,
            )
            normalized = normalized * available.unsqueeze(-1).to(normalized.dtype)
            fused = normalized.sum(dim=1)

        return self.output_norm(self.dropout(fused))

    def _validate_sources(
        self,
        sources: list[torch.Tensor],
    ) -> None:
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


class CrossAttentionFusion(nn.Module):
    def __init__(self, cfg: FusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.source_norms = nn.ModuleList(
            nn.LayerNorm(cfg.emb_dim) for _ in range(cfg.num_sources)
        )
        self.cross_attn = nn.MultiheadAttention(
            cfg.emb_dim,
            cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.output_norm = nn.LayerNorm(cfg.emb_dim)

    def forward(
        self,
        sources: list[torch.Tensor],
        available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_sources(sources)

        normalized = torch.stack(
            [self.source_norms[i](sources[i]) for i in range(self.cfg.num_sources)],
            dim=1,
        )

        if available is not None:
            key_padding_mask = ~available.bool()
        else:
            key_padding_mask = None

        attended, _ = self.cross_attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
        )

        if available is not None:
            attended = attended * available.unsqueeze(-1).to(attended.dtype)
            count = available.float().sum(dim=1, keepdim=True).clamp_min(1.0)
            fused = attended.sum(dim=1) / count
        else:
            fused = attended.mean(dim=1)

        return self.output_norm(self.dropout(fused))

    def _validate_sources(
        self,
        sources: list[torch.Tensor],
    ) -> None:
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


def build_fusion(cfg: FusionConfig, fusion_type: str) -> nn.Module:
    if fusion_type == "gated":
        return MaskedGatedFusion(cfg)
    elif fusion_type == "cross_attention":
        return CrossAttentionFusion(cfg)
    elif fusion_type == "sum":
        return SumFusion(cfg)
    else:
        raise ValueError(f"Unknown fusion_type: {fusion_type}")
