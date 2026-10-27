"""KGAN (Zhang et al., ESWA 2023), adapted to canonical EDuRec queries.

Defaults follow StZHY/KGAN main.py at commit
041f80099ad5232fbb1b04fa0fcad47de4edc407; see the baseline section in README.md.
"""

from collections import defaultdict
from dataclasses import dataclass, replace
from functools import cached_property
from typing import NamedTuple

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from edurec import settings
from edurec.datasets import (
    DataAdapter,
    ElearningDataModule,
    RecSysQuery,
    graphs,
    queries,
)
from edurec.datasets.cache import ProcessedData
from edurec.datasets.queries import QueryConfig
from edurec.evaluation.baselines import register_baseline
from edurec.recsys.configs import TrainConfig
from edurec.recsys.ranking import build_ranking_metrics, update_ranking_metrics
from edurec.recsys.training import train_model


@dataclass
class KGANConfig:
    # Figure 3 peaks at 16; the public main.py defaults to 32.
    emb_dim: int = 16
    n_hops: int = 2
    memory_size: int = 32
    kge_weight: float = 0.01
    l2_weight: float = 1e-6
    candidate_chunk_size: int = 64
    negatives_per_positive: int = 1
    using_all_hops: bool = True

    def __post_init__(self) -> None:
        for name in (
            "emb_dim",
            "n_hops",
            "memory_size",
            "candidate_chunk_size",
            "negatives_per_positive",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.kge_weight < 0 or self.l2_weight < 0:
            raise ValueError("KGAN loss weights must be non-negative.")


def kgan_train_defaults() -> TrainConfig:
    """Published main.py defaults; the evaluation seed/protocol remain shared."""
    return TrainConfig(epochs=20, lr=1e-3, batch_size=1024, weight_decay=0.0)


class KGANInput(NamedTuple):
    query: RecSysQuery
    # Unbatched: [hop, relation, memory]; batched: [batch, hop, relation, memory].
    heads: torch.Tensor
    tails: torch.Tensor
    valid_mask: torch.Tensor


class _KGANDataset(Dataset):
    def __init__(self, adapter: "KGANAdapter", split: str):
        self.adapter = adapter
        self.split = split
        self.queries = adapter.get_queries(split)

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, index: int) -> KGANInput:
        return self.adapter.input_for_query(self.queries[index], self.split)


class KGANAdapter(DataAdapter[KGANInput, KGANInput]):
    """Sample relation-grouped memories from each query's permitted evidence.

    Temporal queries use their strictly-past canonical history. Unordered data
    uses TRAIN positives, excluding the current target during training. No user
    embedding or target-dependent evaluation input is used.
    """

    def __init__(
        self,
        data: ProcessedData,
        config: QueryConfig | None = None,
        model_config: KGANConfig | None = None,
    ):
        self.model_config = model_config or KGANConfig()
        config = replace(
            config or QueryConfig(),
            negatives_per_positive=self.model_config.negatives_per_positive,
        )
        super().__init__(data, config)

    @cached_property
    def graph(self):
        return graphs.build_knowledge_graph(
            self.data.item_features,
            self.data.i_static_feats,
            self.data.schema["items"],
            self.data.graph_node_features,
        )

    @cached_property
    def node_offsets(self) -> dict[str, int]:
        offsets: dict[str, int] = {}
        offset = 0
        for node in ["item", *sorted(n for n in self.graph.node_types if n != "item")]:
            offsets[node] = offset
            offset += int(self.graph[node].num_nodes)
        return offsets

    @property
    def num_entities(self) -> int:
        return sum(int(self.graph[node].num_nodes) for node in self.node_offsets)

    @cached_property
    def relation_types(self) -> list[tuple[str, str, str]]:
        relations = sorted(self.graph.edge_types)
        if not relations:
            raise ValueError(
                "KGAN requires a knowledge graph with static item relations."
            )
        return relations

    @cached_property
    def adjacency(self) -> dict[int, list[tuple[int, int]]]:
        adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for relation, (source, name, target) in enumerate(self.relation_types):
            edges = self.graph[source, name, target].edge_index.t().tolist()
            for head, tail in edges:
                adjacency[head + self.node_offsets[source]].append(
                    (relation, tail + self.node_offsets[target])
                )
        return dict(adjacency)

    @cached_property
    def train_items(self) -> dict[int, list[int]]:
        items: dict[int, list[int]] = defaultdict(list)
        pairs = self.relevant_splits["train"][[settings.USER_COL, settings.ITEM_COL]]
        for user, item in pairs.to_numpy(dtype=np.int64):
            items[int(user)].append(int(item))
        return dict(items)

    def prepare(self, stage: str | None = None) -> None:
        _ = self.adjacency

    def dataset(self, split: str) -> Dataset[KGANInput]:
        return _KGANDataset(self, split)

    def _train_negatives(self, config: QueryConfig) -> np.ndarray:
        # KGAN treats relevant explicit events as positives too, with sampled
        # negatives for binary cross entropy. Keep the shared exclusion policy.
        if config not in self._negative_cache:
            self._negative_cache[config] = queries.generate_negative_samples(
                self.relevant_splits["train"],
                np.arange(self.data.num_items),
                config.negatives_per_positive,
                config.random_state,
                observed_interactions=pd.concat(self.relevant_splits.values()),
            )
        result = self._negative_cache[config]
        assert result is not None
        return result

    def input_for_query(self, query: RecSysQuery, split: str) -> KGANInput:
        if self.data.has_temporal_order:
            # Canonical histories reserve zero for padding; graph IDs do not.
            seeds = (query.history_items[query.history_valid_mask] - 1).tolist()
        else:
            seeds = self.train_items.get(int(query.user_id), [])
            if split == "train":
                seeds = [item for item in seeds if item != int(query.target_item_id)]
        heads, tails, valid = self.sample_memories(seeds)
        return KGANInput(query, heads, tails, valid)

    def evaluation_input(self, query: RecSysQuery) -> KGANInput:
        if query.user_id.ndim == 0:
            return self.input_for_query(query, "test")
        rows = [
            self.input_for_query(RecSysQuery(*(field[i] for field in query)), "test")
            for i in range(len(query.user_id))
        ]
        return KGANInput(
            query,
            torch.stack([row.heads for row in rows]),
            torch.stack([row.tails for row in rows]),
            torch.stack([row.valid_mask for row in rows]),
        )

    def sample_memories(
        self, seeds: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Local deterministic RNG: workers, batch order and model scoring never
        # change the sampled evidence. Equivalent histories get equal memories.
        seed = int(self.config.random_state or 0) % (2**32)
        for item in seeds:
            seed = (seed * 1664525 + int(item) + 1013904223) % (2**32)
        rng = np.random.default_rng(seed)
        shape = (
            self.model_config.n_hops,
            len(self.relation_types),
            self.model_config.memory_size,
        )
        heads = torch.full(shape, self.num_entities, dtype=torch.long)
        tails = heads.clone()
        valid = torch.zeros(shape, dtype=torch.bool)
        frontier = seeds
        for hop in range(self.model_config.n_hops):
            groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
            for head in frontier:
                for relation, tail in self.adjacency.get(head, []):
                    groups[relation].append((head, tail))
            frontier = []
            for relation, triples in groups.items():
                indices = rng.choice(
                    len(triples),
                    self.model_config.memory_size,
                    replace=len(triples) < self.model_config.memory_size,
                )
                sampled = torch.tensor([triples[i] for i in indices])
                heads[hop, relation], tails[hop, relation] = sampled.t()
                valid[hop, relation] = True
                frontier.extend(sampled[:, 1].tolist())
        return heads, tails, valid


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Empty groups have exactly zero weight, including fully empty histories.
    weights = logits.masked_fill(~mask, torch.finfo(logits.dtype).min).softmax(-1)
    weights = weights * mask
    return weights / weights.sum(-1, keepdim=True).clamp_min(
        torch.finfo(logits.dtype).tiny
    )


class KGAN(L.LightningModule):
    """Candidate-conditioned intra/inter-group attention with DistMult.

    Scores are logits in canonical item order, without a padding item column.
    """

    def __init__(
        self,
        num_items: int,
        num_entities: int,
        num_relations: int,
        cfg: KGANConfig | None = None,
        train_cfg: TrainConfig | None = None,
        val_topk: int = settings.TOP_K,
    ):
        super().__init__()
        cfg = cfg or KGANConfig()
        train_cfg = train_cfg or kgan_train_defaults()
        if not 0 < num_items <= num_entities or num_relations < 1:
            raise ValueError("KGAN requires items, entities and at least one relation.")
        self.save_hyperparameters()
        self.cfg, self.train_cfg = cfg, train_cfg
        self.num_items, self.num_entities = num_items, num_entities
        self.val_topk, self.topks = val_topk, sorted(set(train_cfg.topks))
        self.monitor = f"val/ndcg@{val_topk}"
        self.entities = nn.Embedding(
            num_entities + 1, cfg.emb_dim, padding_idx=num_entities
        )
        self.relations = nn.Embedding(num_relations, cfg.emb_dim)
        self.group_attention = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * cfg.emb_dim, cfg.emb_dim),
                    nn.ReLU(),
                    nn.Linear(cfg.emb_dim, 1),
                    nn.ReLU(),
                )
                for _ in range(cfg.n_hops)
            ]
        )
        nn.init.xavier_uniform_(self.entities.weight)
        nn.init.xavier_uniform_(self.relations.weight)
        for module in self.group_attention.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.entities.weight[num_entities].zero_()
        self.val_metrics = build_ranking_metrics([val_topk], "val/")
        self.test_metrics = build_ranking_metrics(
            self.topks, "test/", adaptive_k=train_cfg.adaptive_k
        )

    def forward(
        self, batch: KGANInput, candidate_item_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return [batch, candidates] logits, chunking full-catalog attention."""
        if candidate_item_ids is None:
            candidate_item_ids = torch.arange(self.num_items, device=self.device)
            candidate_item_ids = candidate_item_ids.expand(batch.heads.size(0), -1)
        relations = self.relations.weight
        heads, tails = self.entities(batch.heads), self.entities(batch.tails)
        chunks = []
        for ids in candidate_item_ids.split(self.cfg.candidate_chunk_size, dim=1):
            candidates = self.entities(ids)  # [B, C, D]
            user = torch.zeros_like(candidates)
            for hop, attention in enumerate(self.group_attention):
                # Eq. 7: candidate dot (head * relation), within each group.
                keys = heads[:, hop] * relations[None, :, None, :]
                logits = torch.einsum("bcd,brmd->bcrm", candidates, keys)
                weights = _masked_softmax(logits, batch.valid_mask[:, hop, None])
                groups = torch.einsum("bcrm,brmd->bcrd", weights, tails[:, hop])
                rel = relations[None, None].expand_as(groups)
                logits = attention(torch.cat([groups, rel], dim=-1)).squeeze(-1)
                present = batch.valid_mask[:, hop].any(-1)[:, None]
                weights = _masked_softmax(logits, present)
                if self.cfg.using_all_hops or hop == self.cfg.n_hops - 1:
                    user = user + (groups * weights.unsqueeze(-1)).sum(2)
            chunks.append((user * candidates).sum(-1))  # Eqs. 12-13
        return torch.cat(chunks, dim=1)

    def auxiliary_loss(self, batch: KGANInput) -> torch.Tensor:
        heads, tails = self.entities(batch.heads), self.entities(batch.tails)
        relations = self.relations.weight[None, None, :, None, :]
        valid = batch.valid_mask
        count = valid.sum(dim=(0, 2, 3)).clamp_min(1)
        triple_scores = (heads * relations * tails).sum(-1)
        # Bounded positive-triple objective used in the official implementation.
        kge = -((triple_scores.sigmoid() * valid).sum(dim=(0, 2, 3)) / count).sum()
        l2 = (
            (
                heads.square().sum(-1)
                + tails.square().sum(-1)
                + relations.square().sum(-1)
            )
            * valid
        ).sum(dim=(0, 2, 3)) / count
        l2 = l2.sum()
        return self.cfg.kge_weight * kge + self.cfg.l2_weight * l2

    def training_step(self, batch: KGANInput) -> torch.Tensor:
        query = batch.query
        if query.negative_item_ids.size(1) == 0:
            raise ValueError("KGAN training requires sampled negative items.")
        candidates = torch.cat(
            [query.target_item_id[:, None], query.negative_item_ids], 1
        )
        logits = self(batch, candidates)
        labels = torch.zeros_like(logits)
        labels[:, 0] = 1
        loss = F.binary_cross_entropy_with_logits(logits, labels) + self.auxiliary_loss(
            batch
        )
        self.log("train/Loss", loss, batch_size=len(logits), on_epoch=True)
        return loss

    def _evaluate(self, batch: KGANInput, prefix: str) -> None:
        metrics = self.val_metrics if prefix == "val" else self.test_metrics
        topks = [self.val_topk] if prefix == "val" else self.topks
        query = batch.query
        update_ranking_metrics(
            metrics,
            self(batch),
            query.target_item_id,
            query.query_id,
            query.history_items,
            query.history_valid_mask,
            max_k=max(topks),
        )
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            batch_size=len(query.user_id),
            sync_dist=True,
        )

    def validation_step(self, batch: KGANInput) -> None:
        self._evaluate(batch, "val")

    def test_step(self, batch: KGANInput) -> None:
        self._evaluate(batch, "test")

    def predict_step(self, batch: KGANInput) -> torch.Tensor:
        return self(batch)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # L2 is explicit in the KGAN objective, so no additional weight decay.
        return torch.optim.Adam(self.parameters(), lr=self.train_cfg.lr)


@register_baseline("KGAN", train_defaults=kgan_train_defaults)
def eval_kgan(
    dm: ElearningDataModule,
    train_cfg: TrainConfig | None = None,
    val_topk: int = settings.TOP_K,
    compile: bool = settings.COMPILE_MODEL,
    verbose: bool = False,
    run_name: str | None = None,
    experiment_name: str = settings.EXPERIMENT_NAME,
    cfg: KGANConfig | None = None,
) -> pd.DataFrame:
    train_cfg = train_cfg or kgan_train_defaults()

    # Custom model settings must also govern the adapter's memory sampler.
    def make_adapter(data: ProcessedData, query_config: QueryConfig) -> KGANAdapter:
        return KGANAdapter(data, query_config, cfg)

    view = dm.with_adapter(make_adapter, batch_size=train_cfg.batch_size)
    view.setup("fit")
    adapter = view.require_adapter(KGANAdapter)
    model = KGAN(
        dm.data.num_items,
        adapter.num_entities,
        len(adapter.relation_types),
        adapter.model_config,
        train_cfg,
        val_topk,
    )
    trainer, _, timer = train_model(
        model,
        view,
        debug=False,
        epochs=train_cfg.epochs,
        patience=train_cfg.patience,
        monitor=model.monitor,
        compile=compile,
        verbose=verbose,
        run_name=run_name,
        experiment_name=experiment_name,
    )
    metrics = trainer.test(ckpt_path="best", datamodule=view, weights_only=False)[0]
    return pd.DataFrame(
        [
            {
                "model": "KGAN",
                "seed": settings.state["random_state"],
                **{key.removeprefix("test/"): value for key, value in metrics.items()},
                "training_time_s": timer.time_elapsed("train"),
                "inference_time_s": timer.time_elapsed("test"),
            }
        ]
    )
