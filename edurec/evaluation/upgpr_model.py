from dataclasses import dataclass, field
from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from .. import settings
from .upgpr_graph import KnowledgeGraphData


@dataclass
class UPGPRConfig:
    num_users: int
    num_items: int
    emb_dim: int = 100
    hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    max_acts: int = 50
    state_history: int = 1
    beam_widths: list[int] = field(default_factory=lambda: [10, 3, 1])
    gamma: float = 0.99
    dropout: float = 0.5
    entropy_weight: float = 1e-3
    kg_loss_weight: float = 0.1
    num_neg_samples: int = 5
    kg_sample_size: int = 2048
    lr: float = 1e-4
    weight_decay: float = settings.WEIGHT_DECAY
    topks: list[int] = field(default_factory=lambda: settings.TOP_KS.copy())
    adaptive_k: bool = settings.ADAPTIVE_K


class ActorCriticPolicy(nn.Module):
    """Action-aware actor critic used to navigate a variable-degree KG."""

    def __init__(
        self, state_dim: int, emb_dim: int, hidden_dims: list[int], dropout: float
    ):
        super().__init__()
        layers: list[nn.Module] = []
        current = state_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(current, hidden), nn.ELU(), nn.Dropout(dropout)])
            current = hidden
        self.backbone = nn.Sequential(*layers)
        self.actor = nn.Linear(current, 2 * emb_dim)
        self.critic = nn.Linear(current, 1)
        self.scale = sqrt(2 * emb_dim)

    def forward(
        self, state: torch.Tensor, action_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(state)
        query = self.actor(hidden)
        logits = torch.einsum("bd,bad->ba", query, action_embeddings) / self.scale
        return logits, self.critic(hidden).squeeze(-1)


@dataclass
class _BeamPath:
    nodes: list[int]
    relations: list[int]
    log_probability: torch.Tensor


class UPGPR(nn.Module):
    """Unrestricted Policy-Guided Path Reasoning.

    Entity/relation embeddings are learned with TransE negative sampling. An
    actor-critic policy then learns unrestricted multi-hop paths with the
    binary reward proposed by Frej et al. (2024).
    """

    def __init__(self, cfg: UPGPRConfig, graph: KnowledgeGraphData):
        super().__init__()
        self.cfg = cfg
        self.graph = graph
        self.self_loop_relation = graph.num_relations
        self.entity_embeddings = nn.Embedding(graph.num_nodes, cfg.emb_dim)
        self.relation_embeddings = nn.Embedding(graph.num_relations + 1, cfg.emb_dim)
        nn.init.uniform_(
            self.entity_embeddings.weight, -0.5 / cfg.emb_dim, 0.5 / cfg.emb_dim
        )
        nn.init.uniform_(
            self.relation_embeddings.weight, -0.5 / cfg.emb_dim, 0.5 / cfg.emb_dim
        )
        with torch.no_grad():
            self.relation_embeddings.weight[self.self_loop_relation].zero_()

        state_dim = (2 + 2 * cfg.state_history) * cfg.emb_dim
        self.policy = ActorCriticPolicy(
            state_dim, cfg.emb_dim, cfg.hidden_dims, cfg.dropout
        )
        self.register_buffer("edge_index", graph.edge_index, persistent=False)
        self.register_buffer("edge_type", graph.edge_type, persistent=False)
        self.register_buffer("node_type", graph.node_type, persistent=False)

        self._adjacency: list[list[tuple[int, int]]] = [
            [] for _ in range(graph.num_nodes)
        ]
        for src, dst, rel in zip(
            graph.edge_index[0].tolist(),
            graph.edge_index[1].tolist(),
            graph.edge_type.tolist(),
            strict=True,
        ):
            self._adjacency[src].append((rel, dst))
        for neighbours in self._adjacency:
            neighbours.sort()

        self._known_items: list[set[int]] = [set() for _ in range(cfg.num_users)]
        for src, dst, rel in zip(
            graph.edge_index[0].tolist(),
            graph.edge_index[1].tolist(),
            graph.edge_type.tolist(),
            strict=True,
        ):
            if rel == 0 and src < cfg.num_users:
                self._known_items[src].add(dst - graph.item_offset)

        type_nodes = [
            torch.nonzero(graph.node_type == node_type).flatten()
            for node_type in range(graph.num_node_types)
        ]
        max_type_size = max(nodes.numel() for nodes in type_nodes)
        padded = torch.full((len(type_nodes), max_type_size), -1, dtype=torch.long)
        counts = torch.zeros(len(type_nodes), dtype=torch.long)
        for node_type, nodes in enumerate(type_nodes):
            padded[node_type, : nodes.numel()] = nodes
            counts[node_type] = nodes.numel()
        self.register_buffer("type_nodes", padded, persistent=False)
        self.register_buffer("type_counts", counts, persistent=False)

    def forward(self, user_ids: torch.Tensor, use_paths: bool = True) -> torch.Tensor:
        users = self.entity_embeddings(user_ids)
        enrolled = self.relation_embeddings.weight[0]
        item_ids = torch.arange(
            self.graph.item_offset,
            self.graph.item_offset + self.cfg.num_items,
            device=user_ids.device,
        )
        items = self.entity_embeddings(item_ids)
        scores = torch.matmul(users + enrolled, items.t())
        if not use_paths:
            return scores
        for row, user_id in enumerate(user_ids.tolist()):
            for path in self.beam_search(int(user_id)):
                final_node = path.nodes[-1]
                if self._is_item(final_node):
                    item_id = final_node - self.graph.item_offset
                    scores[row, item_id] += path.log_probability.exp()
        return scores

    def kg_loss(self) -> torch.Tensor:
        assert isinstance(self.edge_type, torch.Tensor)
        assert isinstance(self.edge_index, torch.Tensor)
        assert isinstance(self.type_nodes, torch.Tensor)
        assert isinstance(self.type_counts, torch.Tensor)
        assert isinstance(self.node_type, torch.Tensor)

        num_edges = self.edge_type.numel()
        sample_size = min(self.cfg.kg_sample_size, num_edges)
        edge_ids = torch.randint(
            num_edges, (sample_size,), device=self.edge_type.device
        )
        heads = self.edge_index[0, edge_ids]
        tails = self.edge_index[1, edge_ids]
        relations = self.edge_type[edge_ids]
        head_vec = self.entity_embeddings(heads)
        relation_vec = self.relation_embeddings(relations)
        tail_vec = self.entity_embeddings(tails)
        positive = torch.sum((head_vec + relation_vec) * tail_vec, dim=-1)

        tail_types = self.node_type[tails]
        negative_ids = []
        for node_type in tail_types.tolist():
            count = int(self.type_counts[node_type])
            choices = torch.randint(
                count,
                (self.cfg.num_neg_samples,),
                device=self.edge_type.device,
            )
            negative_ids.append(self.type_nodes[node_type, choices])
        negatives = self.entity_embeddings(torch.stack(negative_ids))
        negative = torch.einsum("bd,bnd->bn", head_vec + relation_vec, negatives)
        return F.softplus(-positive).mean() + F.softplus(negative).mean()

    def policy_loss(self, user_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        paths = [[int(user_id)] for user_id in user_ids.tolist()]
        path_relations: list[list[int]] = [[] for _ in paths]
        log_probs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []

        for hop in range(len(self.cfg.beam_widths)):
            states, action_sets = self._batch_policy_inputs(paths, path_relations)
            logits, value = self.policy(states, action_sets[2])
            logits = logits.masked_fill(~action_sets[3], torch.finfo(logits.dtype).min)
            distribution = Categorical(logits=logits)
            selected = distribution.sample()
            selected_rel = action_sets[0].gather(1, selected[:, None]).squeeze(1)
            selected_dst = action_sets[1].gather(1, selected[:, None]).squeeze(1)
            for path, relations, relation, destination in zip(
                paths,
                path_relations,
                selected_rel.tolist(),
                selected_dst.tolist(),
                strict=True,
            ):
                path.append(destination)
                relations.append(relation)
            log_probs.append(distribution.log_prob(selected))
            values.append(value)
            entropies.append(distribution.entropy())
            rewards.append(
                torch.tensor(
                    [
                        self._binary_reward(int(uid), path, hop + 1)
                        for uid, path in zip(user_ids.tolist(), paths, strict=True)
                    ],
                    device=user_ids.device,
                    dtype=states.dtype,
                )
            )

        returns = rewards[-1]
        discounted: list[torch.Tensor] = [returns]
        for reward in reversed(rewards[:-1]):
            returns = reward + self.cfg.gamma * returns
            discounted.append(returns)
        discounted.reverse()

        actor_loss = torch.stack(
            [
                -(log_prob * (ret - value).detach()).mean()
                for log_prob, value, ret in zip(
                    log_probs, values, discounted, strict=True
                )
            ]
        ).sum()
        critic_loss = torch.stack(
            [
                F.mse_loss(value, ret)
                for value, ret in zip(values, discounted, strict=True)
            ]
        ).sum()
        entropy = torch.stack([value.mean() for value in entropies]).sum()
        loss = actor_loss + critic_loss - self.cfg.entropy_weight * entropy
        return loss, rewards[-1].mean()

    def beam_search(self, user_id: int) -> list[_BeamPath]:
        device = self.entity_embeddings.weight.device
        beams = [
            _BeamPath(
                nodes=[user_id],
                relations=[],
                log_probability=torch.zeros((), device=device),
            )
        ]
        for width in self.cfg.beam_widths:
            expanded: list[_BeamPath] = []
            for beam in beams:
                actions = self._actions(beam.nodes, user_id)
                relations = torch.tensor([a[0] for a in actions], device=device)
                destinations = torch.tensor([a[1] for a in actions], device=device)
                state = self._state(beam.nodes, beam.relations, user_id).unsqueeze(0)
                action_emb = self._action_embeddings(relations, destinations).unsqueeze(
                    0
                )
                logits, _ = self.policy(state, action_emb)
                log_probs = F.log_softmax(logits.squeeze(0), dim=-1)
                keep = min(width, len(actions))
                top_probs, top_indices = torch.topk(log_probs, keep)
                for probability, action_idx in zip(top_probs, top_indices, strict=True):
                    relation, destination = actions[int(action_idx)]
                    expanded.append(
                        _BeamPath(
                            nodes=beam.nodes + [destination],
                            relations=beam.relations + [relation],
                            log_probability=beam.log_probability + probability,
                        )
                    )
            beams = expanded
        return beams

    def _batch_policy_inputs(
        self, paths: list[list[int]], path_relations: list[list[int]]
    ):
        device = self.entity_embeddings.weight.device
        relations_list, destinations_list = [], []
        states = []
        for user_id, path, relations in zip(
            (path[0] for path in paths), paths, path_relations, strict=True
        ):
            actions = self._actions(path, user_id)
            relations_list.append([action[0] for action in actions])
            destinations_list.append([action[1] for action in actions])
            states.append(self._state(path, relations, user_id))
        max_actions = max(len(actions) for actions in relations_list)
        shape = (len(paths), max_actions)
        rel_tensor = torch.full(
            shape, self.self_loop_relation, device=device, dtype=torch.long
        )
        dst_tensor = torch.zeros(shape, device=device, dtype=torch.long)
        mask = torch.zeros(shape, device=device, dtype=torch.bool)
        for row, (relations, destinations) in enumerate(
            zip(relations_list, destinations_list, strict=True)
        ):
            size = len(relations)
            rel_tensor[row, :size] = torch.tensor(relations, device=device)
            dst_tensor[row, :size] = torch.tensor(destinations, device=device)
            mask[row, :size] = True
        action_embeddings = self._action_embeddings(rel_tensor, dst_tensor)
        return torch.stack(states), (
            rel_tensor,
            dst_tensor,
            action_embeddings,
            mask,
        )

    def _actions(self, path: list[int], user_id: int) -> list[tuple[int, int]]:
        current = path[-1]
        actions = [(self.self_loop_relation, current)]
        visited = set(path)
        candidates = [
            (rel, dst) for rel, dst in self._adjacency[current] if dst not in visited
        ]
        if len(candidates) > self.cfg.max_acts:
            with torch.no_grad():
                device = self.entity_embeddings.weight.device
                relations = torch.tensor([a[0] for a in candidates], device=device)
                destinations = torch.tensor([a[1] for a in candidates], device=device)
                user = self.entity_embeddings.weight[user_id]
                scores = torch.sum(
                    (user + self.relation_embeddings(relations))
                    * self.entity_embeddings(destinations),
                    dim=-1,
                )
                indices = torch.topk(scores, self.cfg.max_acts).indices.tolist()
            candidates = [candidates[idx] for idx in indices]
        actions.extend(sorted(candidates))
        return actions

    def _state(
        self, path: list[int], relations: list[int], user_id: int
    ) -> torch.Tensor:
        zero = self.entity_embeddings.weight.new_zeros(self.cfg.emb_dim)
        parts = [
            self.entity_embeddings.weight[user_id],
            self.entity_embeddings.weight[path[-1]],
        ]
        for offset in range(1, self.cfg.state_history + 1):
            if len(path) > offset:
                parts.extend(
                    [
                        self.entity_embeddings.weight[path[-1 - offset]],
                        self.relation_embeddings.weight[relations[-offset]],
                    ]
                )
            else:
                parts.extend([zero, zero])
        return torch.cat(parts)

    def _action_embeddings(
        self, relations: torch.Tensor, destinations: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            [self.relation_embeddings(relations), self.entity_embeddings(destinations)],
            dim=-1,
        )

    def _binary_reward(self, user_id: int, path: list[int], num_hops: int) -> float:
        if num_hops <= 1 or not self._is_item(path[-1]):
            return 0.0
        item_id = path[-1] - self.graph.item_offset
        if item_id not in self._known_items[user_id]:
            return 0.0
        if len(set(path)) <= 3 and any(a == b for a, b in zip(path, path[1:])):
            return 0.0
        return 1.0

    def _is_item(self, node_id: int) -> bool:
        return (
            self.graph.item_offset
            <= node_id
            < self.graph.item_offset + self.cfg.num_items
        )
