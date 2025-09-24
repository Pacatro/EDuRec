import torch
from torch import nn


class NeuralHybrid(nn.Module):
    """
    This is the proposed model for the UCO RecSys, NeuralHybrid is a hybrid recommendation model that combines collaborative filtering (CF)
    and content-based (CB) features using a neural network architecture.

    The model uses user and item embeddings for CF, category embeddings for CB features,
    and incorporates continuous features directly. All features are concatenated and passed
    through a multi-layer perceptron (MLP) to predict the rating.

    Args:
        n_users (int): Number of unique users.
        n_items (int): Number of unique items.
        cat_cardinalities (dict[str, int]): Dictionary mapping categorical feature names to their cardinalities.
        cont_features (list[str]): List of continuous feature names.
        emb_dim (int, optional): Embedding dimension for user/item embeddings. Defaults to 128.
        hidden_dims (list[int], optional): List of hidden layer sizes for the MLP. Defaults to [256, 128, 64, 32, 16].
        dropout (float, optional): Dropout rate for the MLP. Defaults to 0.5.
        min_rating (float, optional): Minimum possible rating. Defaults to 1.0.
        max_rating (float, optional): Maximum possible rating. Defaults to 10.0.

    Forward Input:
        batch (dict[str, torch.Tensor]): Batch dictionary containing user_id, item_id,
            categorical and continuous features.

    Forward Output:
        torch.Tensor: Predicted ratings, clamped between min_rating and max_rating.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        cat_cardinalities: dict[str, int],
        cont_features: list[str],
        emb_dim: int = 128,
        hidden_dims: list[int] = [256, 128, 64, 32, 16],
        dropout: float = 0.5,
        min_rating: float = 1.0,
        max_rating: float = 10.0,
    ):
        super().__init__()

        # CF embeddings
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)

        # User-Item biases
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        # Content Categories embeddings (CB)
        self.cat_embeddings = nn.ModuleDict(
            {
                key: nn.Embedding(card, emb_dim // 2)
                for key, card in cat_cardinalities.items()
            }
        )

        # MLP
        n_cat = len(self.cat_embeddings)
        n_num = len(cont_features)
        mlp_input = emb_dim + n_cat * (emb_dim // 2) + n_num
        layers = []
        for h in hidden_dims:
            layers += [
                nn.Linear(mlp_input, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            mlp_input = h

        layers.append(nn.Linear(mlp_input, 1))
        self.mlp = nn.Sequential(*layers)

        # Hyperparameters
        self.cont_features = cont_features
        self.min_rating = min_rating
        self.max_rating = max_rating

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
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

        num_vecs = [batch[n].unsqueeze(1).float() for n in self.cont_features]
        num_embs = (
            torch.cat(num_vecs, dim=1)
            if num_vecs
            else torch.zeros(u.size(0), 0, device=u_emb.device)
        )

        u_b = self.user_bias(u).squeeze(1)
        i_b = self.item_bias(i).squeeze(1)
        x = torch.cat([i_ui, cat_embs, num_embs], dim=1)
        raw = self.mlp(x).squeeze(1) + u_b + i_b

        return torch.clamp(input=raw, min=self.min_rating, max=self.max_rating)
