"""IDRMI baseline: explainable graph-based course recommendation.

This module implements IDRMI (Yang et al., "An explainable graph-based course
recommendation model based on multiple interest factors", Expert Systems with
Applications 264, 2025) as a self-contained PyTorch Lightning module used by
the benchmark evaluation. It intentionally lives outside ``edurec.recsys`` so
the proposed EDuRec architectures stay separate from the external baselines.

Model summary, adapted to EDuRec's leakage-free split protocol:

* ``NGCF`` propagates messages over the user-item bipartite graph built from the
  TRAIN interactions only, producing user embeddings.
* ``KGCN`` runs relation-aware message passing over EDuRec's heterogeneous item
  knowledge graph, producing item embeddings.
* Three rule-based interest factors: course match (Dice co-enrollment), user
  choice (top-k similar users) and course preference (top-k similar items).
* Final score: ``scale * base * interest`` where ``base = sigmoid(<u, i>)`` and
  ``interest`` is the per-user min-max normalised mean of the three factors.

Every interaction-derived quantity (bipartite graph, factor tables) is computed
from the train split only, so target/validation/test information cannot leak.
The ranking used for the metrics is identical to the one produced by the
paper's ``tanh(base * interest)`` because ``tanh`` and the positive ``scale``
are strictly monotone, while keeping the training cross-entropy well
conditioned.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import lightning as L
import pandas as pd
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.retrieval import RetrievalNormalizedDCG

from edurec import settings
from edurec.datasets import ElearningDataModule, RecSysQuery
from edurec.evaluation.baselines import register_baseline
from edurec.recsys.configs import TrainConfig
from edurec.recsys.ranking import build_ranking_metrics, update_ranking_metrics
from edurec.recsys.training import train_model

EdgeType = tuple[str, str, str]

#: Model label written to the evaluation result files.
IDRMI_LABEL = "IDRMI"

#: Refuse to materialise the dense ``[num_users, num_items]`` factor tables
#: beyond this many elements (~1 GiB in float32) to fail loudly instead of
#: silently exhausting memory.
_MAX_FACTOR_ELEMENTS = 250_000_000


@dataclass
class IDRMIConfig:
    """Hyperparameters of the IDRMI baseline.

    Defaults follow the reference implementation: 32-dimensional embeddings and
    3 NGCF propagation layers, with 6 similar users and 3 similar items for the
    interest factors.
    """

    emb_dim: int = 32
    ngcf_layers: int = 3
    ngcf_dropout: float = 0.1
    kgcn_layers: int = 1
    kgcn_aggregator: str = "mean"
    kgcn_dropout: float = 0.1
    top_users: int = 6
    top_items: int = 3
    #: Strictly positive constant that scales the logits. It does not change the
    #: recommendation ranking but keeps the cross-entropy gradients healthy.
    logit_scale: float = 10.0
    factor_chunk_size: int = 4096


class _NGCF(nn.Module):
    """Neural Graph Collaborative Filtering over the user-item bipartite graph.

    Only the user embeddings are consumed by IDRMI: the item side of the
    propagation is needed to mix user and item signals on the shared adjacency
    but the final item representation comes from the KGCN.
    """

    # Buffers created by ``register_buffer``.
    _adj_indices: torch.Tensor
    _adj_values: torch.Tensor

    def __init__(
        self,
        num_users: int,
        num_items: int,
        emb_dim: int,
        num_layers: int,
        dropout: float,
        norm_adj_indices: torch.Tensor,
        norm_adj_values: torch.Tensor,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers

        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        self.weights = nn.ParameterList(
            nn.Parameter(torch.empty(emb_dim, emb_dim)) for _ in range(num_layers)
        )
        self.biases = nn.ParameterList(
            nn.Parameter(torch.empty(1, emb_dim)) for _ in range(num_layers)
        )
        for weight, bias in zip(self.weights, self.biases, strict=True):
            nn.init.xavier_uniform_(weight)
            nn.init.zeros_(bias)

        self.dropout = nn.Dropout(dropout)

        # Sparse normalised adjacency, rebuilt on the active device on demand.
        self.register_buffer("_adj_indices", norm_adj_indices, persistent=False)
        self.register_buffer("_adj_values", norm_adj_values, persistent=False)
        self._adj_sparse: torch.Tensor | None = None

    def _norm_adj(self) -> torch.Tensor:
        if (
            self._adj_sparse is None
            or self._adj_sparse.device != self._adj_indices.device
        ):
            size = self.num_users + self.num_items
            self._adj_sparse = torch.sparse_coo_tensor(
                self._adj_indices,
                self._adj_values,
                (size, size),
            ).coalesce()
        return self._adj_sparse

    def forward(self) -> torch.Tensor:
        """Return the ``[num_users, emb_dim]`` user embedding table."""
        adjacency = self._norm_adj()
        ego = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        all_embeddings = [ego]

        for layer in range(self.num_layers):
            side = torch.sparse.mm(adjacency, ego)
            ego = side @ self.weights[layer] + self.biases[layer]
            ego = F.leaky_relu(ego, negative_slope=0.2)
            ego = self.dropout(ego)
            ego = F.normalize(ego, p=2, dim=1)
            all_embeddings.append(ego)

        final = torch.stack(all_embeddings, dim=0).sum(dim=0)
        return final[: self.num_users]


class _HeteroKGCN(nn.Module):
    """Relation-aware Knowledge Graph Convolutional Network.

    Runs message passing over EDuRec's heterogeneous, item-only knowledge graph.
    Item nodes combine a learned identifier embedding with a projection of their
    dense features; attribute node types get their own embedding table. Each
    edge type carries a relation vector that modulates the neighbour message.
    """

    def __init__(
        self,
        num_items: int,
        node_counts: dict[str, int],
        edge_types: list[EdgeType],
        item_feat_dim: int,
        cfg: IDRMIConfig,
    ):
        super().__init__()
        self.num_items = num_items
        self.num_layers = cfg.kgcn_layers
        self.aggregator = cfg.kgcn_aggregator
        self.edge_types = list(edge_types)

        emb_dim = cfg.emb_dim
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.xavier_uniform_(self.item_emb.weight)
        self.item_proj = (
            nn.Linear(item_feat_dim, emb_dim) if item_feat_dim > 0 else None
        )

        self.attr_embs = nn.ModuleDict(
            {
                node_type: nn.Embedding(count, emb_dim)
                for node_type, count in node_counts.items()
            }
        )
        for embedding in self.attr_embs.values():
            nn.init.xavier_uniform_(cast(nn.Embedding, embedding).weight)

        self.input_norm = nn.LayerNorm(emb_dim)
        self.relations = nn.ParameterList(
            nn.Parameter(torch.empty(emb_dim)) for _ in self.edge_types
        )
        for relation in self.relations:
            nn.init.normal_(relation, std=0.1)

        node_types = ["item", *node_counts]
        self.self_linears = nn.ModuleDict(
            {node_type: nn.Linear(emb_dim, emb_dim) for node_type in node_types}
        )
        self.neighbor_linears = nn.ModuleDict(
            {node_type: nn.Linear(emb_dim, emb_dim) for node_type in node_types}
        )
        self.dropout = nn.Dropout(cfg.kgcn_dropout)

    def forward(
        self,
        edge_index: dict[EdgeType, torch.Tensor],
        item_feats: torch.Tensor,
    ) -> torch.Tensor:
        item = self.item_emb.weight
        if self.item_proj is not None:
            item = item + self.item_proj(item_feats)

        x: dict[str, torch.Tensor] = {
            node_type: cast(nn.Embedding, embedding).weight
            for node_type, embedding in self.attr_embs.items()
        }
        x["item"] = item
        x = {node_type: self.input_norm(value) for node_type, value in x.items()}

        for _ in range(self.num_layers):
            aggregates = {
                node_type: torch.zeros_like(value) for node_type, value in x.items()
            }
            degrees = {
                node_type: torch.zeros(value.size(0), device=value.device)
                for node_type, value in x.items()
            }

            for edge_idx, (source, relation, target) in enumerate(self.edge_types):
                edges = edge_index.get((source, relation, target))
                if edges is None or edges.numel() == 0:
                    continue
                message = x[target][edges[1]] * self.relations[edge_idx].unsqueeze(0)
                aggregates[source].index_add_(0, edges[0], message)
                degrees[source].index_add_(
                    0, edges[0], torch.ones(edges.size(1), device=edges.device)
                )

            updated: dict[str, torch.Tensor] = {}
            for node_type, value in x.items():
                neighbor = aggregates[node_type]
                if self.aggregator == "mean":
                    neighbor = neighbor / degrees[node_type].clamp(min=1.0).unsqueeze(1)
                hidden = self.self_linears[node_type](value) + self.neighbor_linears[
                    node_type
                ](neighbor)
                updated[node_type] = self.dropout(F.relu(hidden))
            x = updated

        return x["item"]


def _build_bipartite_adjacency(
    interactions: torch.Tensor,
    num_users: int,
    num_items: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-normalised ``D^-1 (A + I)`` over the ``[users; items]`` adjacency.

    ``interactions`` holds the TRAIN ``(user, item)`` pairs as a
    ``[num_edges, 2]`` long tensor. The adapter already provides unique pairs;
    coalescing below makes repeated pairs equivalent anyway.
    """
    size = num_users + num_items
    users = interactions[:, 0]
    items = interactions[:, 1] + num_users

    rows = torch.cat([users, items, torch.arange(size)])
    cols = torch.cat([items, users, torch.arange(size)])
    adjacency = torch.sparse_coo_tensor(
        torch.stack([rows, cols]),
        torch.ones(rows.numel()),
        (size, size),
    ).coalesce()

    indices = adjacency.indices()
    row = indices[0]
    degree = torch.zeros(size).index_add_(0, row, adjacency.values())
    values = adjacency.values() / degree[row].clamp(min=1.0)
    return indices, values


def _build_interest_tables(
    interactions: torch.Tensor,
    num_users: int,
    num_items: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the static rule-based factor tables from TRAIN interactions.

    Returns the binary ``[num_users, num_items]`` interaction matrix ``R`` and
    the course-match table ``CM`` where ``CM[u, c]`` is the mean Dice
    co-enrollment between candidate course ``c`` and every course the user
    interacted with.
    """
    if num_users * num_items > _MAX_FACTOR_ELEMENTS:
        raise RuntimeError(
            "IDRMI factor tables would need "
            f"{num_users * num_items:,} elements, above the "
            f"{_MAX_FACTOR_ELEMENTS:,} safety limit. Reduce the dataset or the "
            "factor-table strategy."
        )

    r = torch.zeros(num_users, num_items, dtype=torch.float32)
    r[interactions[:, 0], interactions[:, 1]] = 1.0

    sparse_r = r.to_sparse()
    user_degree = r.sum(dim=1).clamp(min=1.0)
    item_degree = r.sum(dim=0)

    cooccurrence = torch.sparse.mm(sparse_r.t(), sparse_r).to_dense()
    denominator = item_degree.unsqueeze(0) + item_degree.unsqueeze(1)
    dice = torch.where(
        denominator > 0,
        2.0 * cooccurrence / denominator.clamp(min=1e-8),
        torch.zeros_like(cooccurrence),
    )

    indices = sparse_r.indices()
    scaled_values = sparse_r.values() / user_degree[indices[0]]
    normalized_r = torch.sparse_coo_tensor(indices, scaled_values, sparse_r.size())
    course_match = torch.sparse.mm(normalized_r.coalesce(), dice)
    return r, course_match


@torch.no_grad()
def _topk_cosine(
    embeddings: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k most similar rows by cosine similarity, excluding the row itself."""
    embeddings = F.normalize(embeddings, p=2, dim=1)
    num_rows = embeddings.size(0)
    k = min(top_k, max(num_rows - 1, 1))

    indices = torch.empty(num_rows, k, dtype=torch.long, device=embeddings.device)
    weights = torch.empty(num_rows, k, dtype=embeddings.dtype, device=embeddings.device)

    chunk = max(1, min(1024, num_rows))
    for start in range(0, num_rows, chunk):
        end = min(start + chunk, num_rows)
        similarity = embeddings[start:end] @ embeddings.t()
        self_index = torch.arange(end - start, device=embeddings.device)
        similarity[self_index, self_index + start] = -torch.inf
        values, top_indices = similarity.topk(k, dim=1)
        indices[start:end] = top_indices
        weights[start:end] = values

    return indices, weights


@torch.no_grad()
def _topk_euclidean(
    embeddings: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k most similar rows by ``1 / (euclidean_distance + 1)``."""
    num_rows = embeddings.size(0)
    k = min(top_k, max(num_rows - 1, 1))
    squared = (embeddings * embeddings).sum(dim=1)

    indices = torch.empty(num_rows, k, dtype=torch.long, device=embeddings.device)
    weights = torch.empty(num_rows, k, dtype=embeddings.dtype, device=embeddings.device)

    chunk = max(1, min(1024, num_rows))
    for start in range(0, num_rows, chunk):
        end = min(start + chunk, num_rows)
        distances = (
            (
                squared[start:end].unsqueeze(1)
                + squared.unsqueeze(0)
                - 2.0 * embeddings[start:end] @ embeddings.t()
            )
            .clamp(min=0.0)
            .sqrt()
        )
        similarity = 1.0 / (distances + 1.0)
        self_index = torch.arange(end - start, device=embeddings.device)
        similarity[self_index, self_index + start] = -torch.inf
        values, top_indices = similarity.topk(k, dim=1)
        indices[start:end] = top_indices
        weights[start:end] = values

    return indices, weights


class IDRMIModule(L.LightningModule):
    """Lightning wrapper that trains and evaluates the IDRMI baseline.

    The module consumes the same batches as the proposed models
    (:class:`~edurec.datasets.RecSysQuery`) and reports metrics through the
    shared full-catalog retrieval evaluator, so its numbers are directly
    comparable with EDuRec and the RecBole baselines.
    """

    # Buffers created by ``register_buffer``.
    i_static_feats: torch.Tensor
    _interaction_matrix: torch.Tensor
    _course_match: torch.Tensor

    def __init__(
        self,
        *,
        num_users: int,
        num_items: int,
        i_static_feats: torch.Tensor,
        edge_index: dict[EdgeType, torch.Tensor],
        node_counts: dict[str, int],
        train_interactions: torch.Tensor,
        cfg: IDRMIConfig,
        train_cfg: TrainConfig,
        val_topk: int = settings.TOP_K,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["edge_index", "train_interactions", "i_static_feats"]
        )

        self.num_users = num_users
        self.num_items = num_items
        self.cfg = cfg
        self.train_cfg = train_cfg
        self.val_topk = int(val_topk)
        self.topks = sorted(set(train_cfg.topks or [settings.TOP_K]))
        self.monitor = f"val/ndcg@{self.val_topk}"

        self.register_buffer("i_static_feats", i_static_feats, persistent=False)
        self.item_feat_dim = i_static_feats.size(1)

        adj_indices, adj_values = _build_bipartite_adjacency(
            train_interactions, num_users, num_items
        )
        self.ngcf = _NGCF(
            num_users=num_users,
            num_items=num_items,
            emb_dim=cfg.emb_dim,
            num_layers=cfg.ngcf_layers,
            dropout=cfg.ngcf_dropout,
            norm_adj_indices=adj_indices,
            norm_adj_values=adj_values,
        )

        edge_types = list(edge_index)
        self.kgcn = _HeteroKGCN(
            num_items=num_items,
            node_counts=node_counts,
            edge_types=edge_types,
            item_feat_dim=self.item_feat_dim,
            cfg=cfg,
        )
        self._edge_types = edge_types
        for idx, edge_type in enumerate(edge_types):
            self.register_buffer(
                f"edge_index_{idx}", edge_index[edge_type], persistent=False
            )

        r, course_match = _build_interest_tables(
            train_interactions, num_users, num_items
        )
        self.register_buffer("_interaction_matrix", r, persistent=False)
        self.register_buffer("_course_match", course_match, persistent=False)

        self._user_topk_idx: torch.Tensor | None = None
        self._user_topk_weight: torch.Tensor | None = None
        self._item_topk_idx: torch.Tensor | None = None
        self._item_topk_weight: torch.Tensor | None = None
        self._item_emb_cache: torch.Tensor | None = None

        self.val_ranking_metrics = MetricCollection(
            {
                f"ndcg@{self.val_topk}": RetrievalNormalizedDCG(
                    top_k=self.val_topk,
                    empty_target_action="neg",
                    aggregation="mean",
                )
            },
            prefix="val/",
        )
        self.test_ranking_metrics = build_ranking_metrics(
            self.topks,
            "test/",
            adaptive_k=train_cfg.adaptive_k,
        )

    # ------------------------------------------------------------------
    # Graph encoders
    # ------------------------------------------------------------------
    @property
    def edge_index(self) -> dict[EdgeType, torch.Tensor]:
        return {
            edge_type: self.get_buffer(f"edge_index_{idx}")
            for idx, edge_type in enumerate(self._edge_types)
        }

    def _item_embeddings(self) -> torch.Tensor:
        if not self.training and self._item_emb_cache is not None:
            return self._item_emb_cache

        feats = self.i_static_feats
        embeddings = self.kgcn(self.edge_index, feats)
        if not self.training:
            self._item_emb_cache = embeddings
        return embeddings

    def train(self, mode: bool = True) -> "IDRMIModule":
        self._item_emb_cache = None
        return super().train(mode)

    # ------------------------------------------------------------------
    # Interest factors
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _refresh_embedding_factors(self) -> None:
        """Recompute the embedding-dependent factor tables (detached)."""
        # Force a fresh item table: the best checkpoint may have replaced the
        # weights since the previous refresh (e.g. between validation and test).
        self._item_emb_cache = None
        user_embeddings = self.ngcf().detach()
        item_embeddings = self._item_embeddings().detach()
        self._user_topk_idx, self._user_topk_weight = _topk_cosine(
            user_embeddings, self.cfg.top_users
        )
        self._item_topk_idx, self._item_topk_weight = _topk_euclidean(
            item_embeddings, self.cfg.top_items
        )

    def _mix(
        self, user_ids: torch.Tensor, candidate_item_ids: torch.Tensor
    ) -> torch.Tensor:
        """Interest factors for ``(user, candidate)`` pairs, shape ``[batch, k]``."""
        if self._user_topk_idx is None or self._item_topk_idx is None:
            self._refresh_embedding_factors()

        assert self._user_topk_idx is not None
        assert self._user_topk_weight is not None
        assert self._item_topk_idx is not None
        assert self._item_topk_weight is not None

        batch_size = user_ids.size(0)
        num_candidates = candidate_item_ids.size(0)

        course_match = self._course_match[
            user_ids.unsqueeze(1), candidate_item_ids.unsqueeze(0)
        ]

        similar_users = self._user_topk_idx[user_ids]
        similar_user_weight = self._user_topk_weight[user_ids]
        similar_user_r = self._interaction_matrix[similar_users]
        choice_full = (similar_user_r * similar_user_weight.unsqueeze(-1)).sum(dim=1)
        user_choice = choice_full[:, candidate_item_ids] / similar_users.size(1)

        similar_items = self._item_topk_idx[candidate_item_ids]
        similar_item_weight = self._item_topk_weight[candidate_item_ids]
        interaction_rows = self._interaction_matrix[user_ids]
        membership = interaction_rows.gather(
            1,
            similar_items.reshape(1, -1).expand(batch_size, -1),
        ).view(batch_size, num_candidates, similar_items.size(1))
        course_preference = (membership * similar_item_weight.unsqueeze(0)).sum(
            dim=-1
        ) / similar_items.size(1)

        return (course_match + user_choice + course_preference) / 3.0

    @staticmethod
    def _normalize_mix(mix: torch.Tensor) -> torch.Tensor:
        """Per-user min-max normalisation of the factors to ``[0.5, 1.5]``."""
        minimum = mix.min(dim=1, keepdim=True).values
        maximum = mix.max(dim=1, keepdim=True).values
        span = maximum - minimum
        normalized = 0.5 + (mix - minimum) / span.clamp(min=1e-6)
        return torch.where(span > 1e-6, normalized, torch.ones_like(mix))

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _collaborative(
        self, user_ids: torch.Tensor, candidate_item_ids: torch.Tensor
    ) -> torch.Tensor:
        user_embeddings = self.ngcf()
        item_embeddings = self._item_embeddings()

        user = user_embeddings[user_ids]
        chunk = self.cfg.factor_chunk_size
        dot_products = []
        for start in range(0, candidate_item_ids.numel(), chunk):
            items = item_embeddings[candidate_item_ids[start : start + chunk]]
            dot_products.append((user.unsqueeze(1) * items.unsqueeze(0)).sum(dim=-1))
        return torch.sigmoid(torch.cat(dot_products, dim=1))

    def score(
        self, user_ids: torch.Tensor, candidate_item_ids: torch.Tensor
    ) -> torch.Tensor:
        """Final IDRMI scores for the given ``(user, candidate)`` pairs.

        The interest factors are always normalised over the full catalogue, so
        a candidate receives the same interest value regardless of which other
        candidates are scored with it. Training candidate sets and the
        full-catalogue evaluation therefore rank consistently.
        """
        all_candidates = torch.arange(self.num_items, device=user_ids.device)
        interest = self._normalize_mix(self._mix(user_ids, all_candidates))
        interest = interest[:, candidate_item_ids]
        base = self._collaborative(user_ids, candidate_item_ids)
        return self.cfg.logit_scale * base * interest

    def forward(
        self,
        batch: RecSysQuery,
        candidate_item_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        user_ids = batch.user_id.long()
        if candidate_item_ids is None:
            candidate_item_ids = torch.arange(self.num_items, device=user_ids.device)
        else:
            candidate_item_ids = candidate_item_ids.reshape(-1).long()
        return self.score(user_ids, candidate_item_ids)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def on_train_epoch_start(self) -> None:
        self._refresh_embedding_factors()

    def on_validation_epoch_start(self) -> None:
        self._refresh_embedding_factors()

    def on_test_epoch_start(self) -> None:
        self._refresh_embedding_factors()

    def training_step(self, batch: RecSysQuery) -> torch.Tensor:
        return self._step(batch, prefix="train")

    def validation_step(self, batch: RecSysQuery) -> None:
        self._step(
            batch,
            prefix="val",
            ranking_metrics=self.val_ranking_metrics,
            metric_topks=[self.val_topk],
        )

    def test_step(self, batch: RecSysQuery) -> None:
        self._step(
            batch,
            prefix="test",
            ranking_metrics=self.test_ranking_metrics,
            metric_topks=self.topks,
        )

    def _step(
        self,
        batch: RecSysQuery,
        prefix: str,
        ranking_metrics: MetricCollection | None = None,
        metric_topks: list[int] | None = None,
    ) -> torch.Tensor:
        negative_item_ids = batch.negative_item_ids if prefix == "train" else None

        if negative_item_ids is not None and negative_item_ids.size(1) > 0:
            candidate_item_ids = torch.cat(
                [batch.target_item_id.reshape(-1, 1).long(), negative_item_ids],
                dim=1,
            )
            scores = self(batch, candidate_item_ids=candidate_item_ids)
            labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
            rec_loss = F.cross_entropy(scores, labels)
        else:
            scores = self(batch)
            rec_loss = F.cross_entropy(scores, batch.target_item_id.reshape(-1).long())

        self.log(
            f"{prefix}/Loss",
            rec_loss.detach(),
            on_step=prefix == "train",
            on_epoch=prefix != "train",
            prog_bar=prefix == "train",
            logger=True,
            sync_dist=True,
            batch_size=scores.size(0),
        )

        if ranking_metrics is not None and metric_topks:
            update_ranking_metrics(
                metrics=ranking_metrics,
                scores=scores,
                target_item_ids=batch.target_item_id,
                query_ids=batch.query_id,
                history_items=batch.history_items,
                history_mask=batch.history_valid_mask,
                max_k=max(metric_topks),
            )
            self.log_dict(
                ranking_metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=scores.size(0),
            )

        return rec_loss

    def predict_step(self, batch: RecSysQuery) -> torch.Tensor:
        return self(batch)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.train_cfg.lr,
            weight_decay=self.train_cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min" if self.monitor.lower().endswith("loss") else "max",
            factor=0.5,
            patience=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor},
        }


def _train_interactions(dm: ElearningDataModule) -> torch.Tensor:
    """TRAIN ``(user, item)`` pairs as a ``[num_edges, 2]`` long tensor.

    The adapter returns the bimodal TRAIN-only ``edge_index``; IDRMI applies its
    own normalisation and factor tables, which stay baseline-specific.
    """
    edge_index = dm.adapter.get_bipartite_graph()

    return edge_index.t().contiguous().long()


@register_baseline(IDRMI_LABEL)
def eval_idrmi(
    dm: ElearningDataModule,
    train_cfg: TrainConfig | None = None,
    val_topk: int = settings.TOP_K,
    cfg: IDRMIConfig | None = None,
    compile: bool = settings.COMPILE_MODEL,
    results_path: Path | None = None,
    verbose: bool = False,
    run_name: str | None = None,
    experiment_name: str = settings.EXPERIMENT_NAME,
) -> pd.DataFrame:
    """Train and evaluate the IDRMI baseline on ``dm``.

    Mirrors :func:`edurec.evaluation.eval_model` and returns a one-row
    ``DataFrame`` with the ``test/`` metrics so the CLI can persist it in the
    same layout as EDuRec and the RecBole baselines.
    """
    resolved_cfg = cfg or IDRMIConfig()
    resolved_train = train_cfg or TrainConfig()
    dm.setup("fit")

    knowledge_graph = dm.adapter.get_knowledge_graph()
    module = IDRMIModule(
        num_users=dm.num_users,
        num_items=dm.num_items,
        i_static_feats=dm.i_static_feats,
        edge_index={
            edge_type: knowledge_graph[edge_type].edge_index
            for edge_type in knowledge_graph.edge_types
        },
        node_counts=dm.kg_node_counts,
        train_interactions=_train_interactions(dm),
        cfg=resolved_cfg,
        train_cfg=resolved_train,
        val_topk=val_topk,
    )

    trainer, _, timer = train_model(
        model=module,
        dm=dm,
        debug=False,
        epochs=resolved_train.epochs,
        patience=resolved_train.patience,
        monitor=module.monitor,
        compile=compile,
        verbose=verbose,
        run_name=run_name,
        experiment_name=experiment_name,
    )

    metrics = trainer.test(ckpt_path="best", datamodule=dm, weights_only=False)[0]
    metrics = {name.removeprefix("test/"): value for name, value in metrics.items()}
    results = pd.DataFrame(
        [
            {
                "model": IDRMI_LABEL,
                **metrics,
                "training_time_s": timer.time_elapsed("train"),
                "inference_time_s": timer.time_elapsed("test"),
            }
        ]
    )

    if results_path is not None:
        results.to_csv(results_path / f"{IDRMI_LABEL}.csv", index=True)

    return results


__all__ = ["IDRMI_LABEL", "IDRMIConfig", "IDRMIModule", "eval_idrmi"]
