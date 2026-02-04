from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics import MetricCollection

from .. import config


@dataclass
class EDuRecConfig:
    """
    Configuration dataclass for the EDuRec model.

    Attributes:
        n_users (int): Number of unique users.
        n_items (int): Number of unique items.
        cat_cardinalities (dict[str, int]): Dictionary mapping categorical feature names to their cardinalities.
        numeric_features (list[str]): List of continuous feature names.
        emb_dim (int, optional): Embedding dimension for user/item embeddings. Defaults to 128.
        hidden_dims (list[int], optional): List of hidden layer sizes for the MLP. Defaults to [256, 128, 64, 32, 16].
        dropout (float, optional): Dropout rate for the MLP. Defaults to 0.5.
    """

    n_users: int
    n_items: int
    cat_cardinalities: dict[str, int]
    numeric_features: list[str]
    emb_dim: int = config.EMB_DIM
    hidden_dims: list[int] = field(default_factory=lambda: config.HIDDEN_DIMS)
    dropout: float = config.DROPOUT


class EDuRecMTL(nn.Module):
    def __init__(self, config: EDuRecConfig):
        super().__init__()
        self.config = config

        # Shared Bottom
        self.user_embedding = nn.Embedding(config.n_users, config.emb_dim)
        self.item_embedding = nn.Embedding(config.n_items, config.emb_dim)

        cat_emb_dim = config.emb_dim // 2

        self.cat_embeddings = nn.ModuleDict(
            {
                key: nn.Embedding(card, cat_emb_dim)
                for key, card in config.cat_cardinalities.items()
            }
        )

        # Cross-attention user-item interactions
        self.ui_interactions = CrossAttentionInteraction(config.emb_dim)

        # Common MLP
        n_cat = len(self.cat_embeddings)
        n_num = len(config.numeric_features)
        mlp_input = (config.emb_dim * 3) + n_cat * cat_emb_dim + n_num

        shared_layers = []

        for h in config.hidden_dims:
            shared_layers += [
                nn.Linear(mlp_input, h),
                nn.LayerNorm(h),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout),
            ]
            mlp_input = h

        self.shared_mlp = nn.Sequential(*shared_layers)

        # Rating Head
        self.rating_head = nn.Linear(config.hidden_dims[-1], 1)
        self.user_bias = nn.Embedding(config.n_users, 1)
        self.item_bias = nn.Embedding(config.n_items, 1)

        # Relevance Head
        self.relevance_head = nn.Linear(config.hidden_dims[-1], 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        u = batch["user_id"].long()
        i = batch["item_id"].long()

        # Shared embeddings
        u_emb = self.user_embedding(u)
        i_emb = self.item_embedding(i)
        ui_i = self.ui_interactions(u_emb, i_emb)

        cat_vecs = [emb(batch[key].long()) for key, emb in self.cat_embeddings.items()]
        cat_embs = (
            torch.cat(cat_vecs, dim=1)
            if cat_vecs
            else torch.zeros(u.size(0), 0, device=u_emb.device)
        )

        num_vecs = [batch[n].unsqueeze(1).float() for n in self.config.numeric_features]
        num_embs = (
            torch.cat(num_vecs, dim=1)
            if num_vecs
            else torch.zeros(u.size(0), 0, device=u_emb.device)
        )

        x = torch.cat([u_emb, i_emb, ui_i, cat_embs, num_embs], dim=1)
        shared_out = self.shared_mlp(x)

        # Rating prediction
        u_b = self.user_bias(u).squeeze(1)
        i_b = self.item_bias(i).squeeze(1)
        rating_out = self.rating_head(shared_out).squeeze(1) + u_b + i_b

        # Relevance prediction
        relevance_out = self.relevance_head(shared_out).squeeze(1)

        return {
            "rating": rating_out,
            "relevance": relevance_out,
        }

    def compute_loss(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        alpha: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        pred_ratings = preds["rating"].flatten()
        logits_relevance = preds["relevance"].flatten()
        true_ratings = batch[config.RATING_COL].float().view_as(pred_ratings)
        true_relevance = batch[config.RELEVANT_COL].float().view_as(logits_relevance)

        loss_rating = F.mse_loss(pred_ratings, true_ratings)
        loss_relevance = F.binary_cross_entropy_with_logits(
            logits_relevance, true_relevance
        )

        loss = (alpha * loss_rating) + ((1 - alpha) * loss_relevance)

        logs = {
            "LossRating": loss_rating,
            "LossRelevance": loss_relevance,
        }

        return loss, logs

    def compute_ranking_metrics(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        ranking_metrics: MetricCollection,
    ) -> None:
        user_ids = batch[config.USER_COL].long().flatten()
        target = batch[config.RELEVANT_COL].bool().flatten()
        ranking_preds = preds["relevance"].detach()
        ranking_metrics.update(ranking_preds, target, indexes=user_ids)


class CrossAttentionInteraction(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, u_emb: torch.Tensor, i_emb: torch.Tensor) -> torch.Tensor:
        # Expand dim (batch, emb_dim) to (batch, 1, emb_dim)
        q = u_emb.unsqueeze(1)
        k = i_emb.unsqueeze(1)
        v = i_emb.unsqueeze(1)

        attn_out, _ = self.attn(q, k, v)  # (batch, 1, emb_dim)
        out = self.norm(attn_out.squeeze(1) + u_emb)  # (batch, emb_dim)
        return out


class EDuRecV1(nn.Module):
    """
    This is the proposed model for the RecSys, is a hybrid recommendation model that combines collaborative filtering (CF)
    and content-based (CB) features using a neural network architecture.

    The model uses user and item embeddings for CF, category embeddings for CB features,
    and incorporates continuous features directly. All features are concatenated and passed
    through a multi-layer perceptron (MLP) to predict the rating.

    Args:
        config (EDuRecConfig): Configuration dataclass for the EDuRec model.

    Forward Input:
        batch (dict[str, torch.Tensor]): Batch dictionary containing user_id, item_id,
            categorical and continuous features.

    Forward Output:
        torch.Tensor: Predicted ratings, clamped between min_rating and max_rating.
    """

    def __init__(self, config: EDuRecConfig):
        super().__init__()

        self.config = config

        # CF embeddings
        self.user_embedding = nn.Embedding(config.n_users, config.emb_dim)
        self.item_embedding = nn.Embedding(config.n_items, config.emb_dim)

        # User-Item biases
        self.user_bias = nn.Embedding(config.n_users, 1)
        self.item_bias = nn.Embedding(config.n_items, 1)

        # Content Categories embeddings (CB)
        cat_emb_dim = config.emb_dim // 2
        self.cat_embeddings = nn.ModuleDict(
            {
                key: nn.Embedding(card, cat_emb_dim)
                for key, card in config.cat_cardinalities.items()
            }
        )

        # MLP
        n_cat = len(self.cat_embeddings)
        n_num = len(config.numeric_features)
        mlp_input = config.emb_dim + n_cat * (cat_emb_dim) + n_num
        layers = []
        for h in config.hidden_dims:
            layers += [
                nn.Linear(mlp_input, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout),
            ]
            mlp_input = h

        layers.append(nn.Linear(mlp_input, 1))
        self.mlp = nn.Sequential(*layers)

        # Hyperparameters
        self.numeric_features = config.numeric_features

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        u = batch["user_id"].long()
        i = batch["item_id"].long()

        u_emb = self.user_embedding(u)
        i_emb = self.item_embedding(i)

        i_ui = u_emb * i_emb

        cat_vecs = [emb(batch[key].long()) for key, emb in self.cat_embeddings.items()]
        cat_embs = (
            torch.cat(cat_vecs, dim=1)
            if cat_vecs
            else torch.zeros(u.size(0), 0, device=u_emb.device)
        )

        num_vecs = [batch[n].unsqueeze(1).float() for n in self.numeric_features]
        num_embs = (
            torch.cat(num_vecs, dim=1)
            if num_vecs
            else torch.zeros(u.size(0), 0, device=u_emb.device)
        )

        u_b = self.user_bias(u).squeeze(1)
        i_b = self.item_bias(i).squeeze(1)

        x = torch.cat([i_ui, cat_embs, num_embs], dim=1)

        out = self.mlp(x).squeeze(1) + u_b + i_b

        return {"rating": out}

    def compute_loss(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        _ = args, kwargs
        loss = F.mse_loss(preds["rating"], batch[config.RATING_COL])
        return loss, None

    def compute_ranking_metrics(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        ranking_metrics: MetricCollection,
    ) -> None:
        user_ids = batch[config.USER_COL].long()
        target = batch[config.RELEVANT_COL].long()
        ranking_preds = preds["rating"].detach()
        ranking_metrics.update(ranking_preds, target, indexes=user_ids)
