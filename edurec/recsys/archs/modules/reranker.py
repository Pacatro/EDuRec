from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class RerankerConfig:
    emb_dim: int
    hidden_dims: list[int] = field(default_factory=list)
    dropout: float = 0.1


class CandidateReranker(nn.Module):
    """MLP that scores each candidate from contextual features.

    For every candidate it consumes the item representation, the history
    context produced by candidate attention and the (profile-fused) user
    representation, then returns one logit per candidate.
    """

    def __init__(self, cfg: RerankerConfig):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = cfg.emb_dim * 3

        for hidden_dim in cfg.hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        candidate_emb: torch.Tensor,
        context: torch.Tensor,
        user_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidates.

        Args:
            candidate_emb: Item representations ``[batch, candidates, dim]``.
            context: History-context representations ``[batch, candidates, dim]``.
            user_emb: User representation ``[batch, dim]``.
        Returns:
            Logits with shape ``[batch, candidates]``.
        """
        batch_size, num_candidates, _ = candidate_emb.shape
        user = user_emb.unsqueeze(1).expand(batch_size, num_candidates, -1)
        joint = torch.cat([candidate_emb, context, user], dim=-1)
        return self.mlp(joint).squeeze(-1)
