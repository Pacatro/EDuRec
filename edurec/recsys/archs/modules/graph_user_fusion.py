import torch
from torch import nn


class GraphUserFusion(nn.Module):
    """Fuse sequential and graph users, ignoring unavailable graph signals."""

    def __init__(self, emb_dim: int):
        super().__init__()
        self.gate = nn.Linear(emb_dim * 2, emb_dim)

    def forward(
        self,
        sequential: torch.Tensor,
        graph_users: torch.Tensor,
        user_ids: torch.Tensor,
        has_history: torch.Tensor,
        edge_index: dict[tuple[str, str, str], torch.Tensor],
        enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        connected = torch.zeros(
            graph_users.size(0), dtype=torch.bool, device=graph_users.device
        )
        if enabled:
            for (_, _, target), edges in edge_index.items():
                if target == "user":
                    connected[edges[1]] = True
        ids = user_ids.reshape(-1)
        valid = (ids >= 0) & (ids < graph_users.size(0))
        # An extra zero row handles unknown IDs and empty user tables.
        padded = torch.cat([graph_users, graph_users.new_zeros(1, graph_users.size(1))])
        safe_ids = torch.where(valid, ids, graph_users.size(0))
        available = torch.cat([connected, connected.new_zeros(1)])[safe_ids]
        graph = padded[safe_ids]
        gate = torch.sigmoid(self.gate(torch.cat([sequential, graph], dim=-1)))
        fused = gate * sequential + (1.0 - gate) * graph
        fused = torch.where(has_history[:, None], fused, graph)
        return torch.where(available[:, None], fused, sequential), available
