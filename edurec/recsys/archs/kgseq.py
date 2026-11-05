import torch
from torch import nn

from edurec.recsys.archs.base import BaseRecArch
from edurec.recsys.archs.modules import (
    FeatureEncoder,
    GraphEncoder,
    Scorer,
    SeqEncoder,
    UserStateEncoder,
    clamp_user_ids,
    drop_relation_edges,
    item_contrastive_loss,
)
from edurec.recsys.configs import ModelConfig


class KGSeq(BaseRecArch):
    """Item-knowledge-graph educational recommender with a recurrent encoder.

    Item relations are encoded by a relation-aware item GNN that never sees
    user-item interactions. Each historical event is the sum of its item
    embedding, its interaction context and its time gap:

    ``x_t = LayerNorm(item_emb[t] + context[t] + time[t])``

    The static user profile initialises the recurrent state
    (``h0 = P_user(profile)``), the GRU/LSTM produces the user state (last valid
    hidden state by default), and an MLP scores user/candidate pairs.

    Optional GCL regularizes this same graph encoder using two independently
    edge-dropped views. Recommendation always uses the complete knowledge
    graph; the augmented embeddings only enter the auxiliary training loss.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        user_dense_dim = cfg.num_user_dense_feats if cfg.use_user_features else 0
        user_cats = cfg.user_cat_cardinalities if cfg.use_user_features else []
        use_user_ids = cfg.use_user_id_embedding and cfg.use_user_features

        edge_types = [
            (source, relation, target) for source, relation, target in cfg.kg_edge_types
        ]
        self.kg = GraphEncoder(
            num_items=cfg.num_items,
            emb_dim=cfg.emb_dim,
            item_feat_dim=cfg.effective_item_dense_feats,
            num_layers=cfg.gnn_layers,
            heads=cfg.gnn_heads,
            dropout=cfg.gnn_dropout,
            node_counts=dict(cfg.kg_node_counts),
            node_feat_dims=(
                dict(cfg.kg_node_feat_dims)
                if cfg.use_text_features and cfg.graph_mode == "kg"
                else {}
            ),
            edge_types=edge_types,
            graph_mode=cfg.graph_mode,
        )
        self.interaction_context = (
            FeatureEncoder(
                emb_dim=cfg.emb_dim,
                dense_dim=cfg.num_interaction_dense_feats,
                cat_cardinalities=cfg.interaction_cat_cardinalities,
            )
            if cfg.has_interaction_context
            else None
        )
        self.user_state = (
            UserStateEncoder(
                emb_dim=cfg.emb_dim,
                hidden_dim=cfg.rnn_hidden_dim,
                num_layers=cfg.rnn_layers,
                num_dense_feats=user_dense_dim,
                cat_cardinalities=user_cats,
                num_users=cfg.num_users,
                use_id_embedding=use_user_ids,
            )
            if cfg.has_history
            else None
        )
        self.time_proj = (
            nn.Linear(1, cfg.emb_dim)
            if cfg.use_time_features and cfg.has_history
            else None
        )
        self.event_norm = nn.LayerNorm(cfg.emb_dim)

        self.seq_encoder = (
            SeqEncoder(
                emb_dim=cfg.emb_dim,
                hidden_dim=cfg.rnn_hidden_dim,
                num_layers=cfg.rnn_layers,
                dropout=cfg.rnn_dropout,
                cell_type=cfg.rnn_type,
            )
            if cfg.has_history
            else None
        )
        self.static_user = (
            FeatureEncoder(
                emb_dim=cfg.emb_dim,
                dense_dim=user_dense_dim,
                cat_cardinalities=user_cats,
                num_ids=cfg.num_users if use_user_ids else 0,
            )
            if not cfg.has_history and cfg.has_user_profile
            else None
        )
        self.user_embedding = (
            nn.Embedding(cfg.num_users, cfg.emb_dim) if not cfg.has_history else None
        )

        self.item_bias = (
            nn.Parameter(torch.zeros(cfg.num_items)) if cfg.use_item_bias else None
        )
        self.scorer = Scorer(
            emb_dim=cfg.emb_dim,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
            max_pairs_per_chunk=cfg.scorer_max_pairs_per_chunk,
        )

    def _compute_item_embeddings(self) -> torch.Tensor:
        return self.kg(self.edge_index, self.item_dense_feats, self.graph_node_feats)

    def auxiliary_loss(
        self,
        h_ids: torch.Tensor,
        h_mask: torch.Tensor,
        target_item_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        """Unweighted GCL loss on training history items and targets.

        History IDs are shifted by one (zero is padding), while target IDs
        already index the item table. Targets select contrastive anchors only;
        they do not become history events or graph edges.
        """
        if not self.training or not self.cfg.gcl_enabled:
            return None

        ids = torch.cat(
            [h_ids[h_mask.bool()].long() - 1, target_item_ids.reshape(-1).long()]
        ).unique()
        if ids.numel() > self.cfg.gcl_max_items:
            selected = torch.randperm(ids.numel(), device=ids.device)
            ids = ids[selected[: self.cfg.gcl_max_items]]

        item_feats = self.item_dense_feats
        edges = self.edge_index
        first = self.kg(
            drop_relation_edges(edges, self.cfg.gcl_edge_dropout),
            item_feats,
            self.graph_node_feats,
        )
        second = self.kg(
            drop_relation_edges(edges, self.cfg.gcl_edge_dropout),
            item_feats,
            self.graph_node_feats,
        )
        return item_contrastive_loss(first, second, ids, self.cfg.gcl_temperature)

    def forward(
        self,
        h_ids: torch.Tensor,
        h_mask: torch.Tensor,
        user_ids: torch.Tensor,
        h_dense: torch.Tensor | None = None,
        h_cat: torch.Tensor | None = None,
        h_delta: torch.Tensor | None = None,
        candidate_item_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        item_emb = self.item_embeddings()
        user_emb = self._encode_user(
            item_emb, h_ids, h_mask, user_ids, h_dense, h_cat, h_delta
        )

        scores = self.scorer(user_emb, item_emb, item_ids=candidate_item_ids)

        if self.item_bias is not None:
            bias = (
                self.item_bias
                if candidate_item_ids is None
                else self.item_bias[candidate_item_ids]
            )
            scores = scores + bias

        return scores

    def _encode_user(
        self,
        item_emb: torch.Tensor,
        h_ids: torch.Tensor,
        h_mask: torch.Tensor,
        user_ids: torch.Tensor,
        h_dense: torch.Tensor | None,
        h_cat: torch.Tensor | None,
        h_delta: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.seq_encoder is None:
            return self._static_user(user_ids)

        padded = torch.cat([item_emb.new_zeros(1, item_emb.size(1)), item_emb])
        event = padded[h_ids.clamp(min=0)]

        if (
            self.interaction_context is not None
            and h_dense is not None
            and h_cat is not None
        ):
            event = event + self.interaction_context(h_dense, h_cat)

        if self.time_proj is not None and h_delta is not None:
            event = event + self.time_proj(
                torch.log1p(h_delta.clamp(min=0.0)).unsqueeze(-1)
            )

        event = self.event_norm(event)

        assert self.user_state is not None
        initial_state = self.user_state(
            self.u_cat_feats, self.u_static_feats, user_ids
        )
        return self.seq_encoder(event, h_mask, initial_state=initial_state)

    def _static_user(self, user_ids: torch.Tensor) -> torch.Tensor:
        assert self.user_embedding is not None
        user = self.user_embedding(user_ids)
        if self.static_user is not None:
            ids = clamp_user_ids(user_ids, self.cfg.num_users)
            user = user + self.static_user(
                self.u_static_feats[ids], self.u_cat_feats[ids], ids=ids
            )
        return user
