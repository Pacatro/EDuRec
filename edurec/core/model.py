import torch
from torch import nn


class EDuRec(nn.Module):
    """
    This is the proposed model for the RecSys, is a hybrid recommendation model that combines collaborative filtering (CF)
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
        **kwargs,
    ):
        super().__init__()
        _ = kwargs

        # CF embeddings
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)

        # User-Item biases
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        # Content Categories embeddings (CB)
        cat_emb_dim = emb_dim // 2
        self.cat_embeddings = nn.ModuleDict(
            {
                key: nn.Embedding(card, cat_emb_dim)
                for key, card in cat_cardinalities.items()
            }
        )

        # MLP
        n_cat = len(self.cat_embeddings)
        n_num = len(cont_features)
        mlp_input = emb_dim + n_cat * (cat_emb_dim) + n_num
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
        self.cont_features: list[str] = cont_features

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

        # return torch.clamp(input=raw, min=self.min_rating, max=self.max_rating)
        return raw


class MF(nn.Module):
    """
    A simple implementation of the Matrix Factorization model.

    The model factorizes the user-item interaction matrix into
    the product of two lower-rank matrices, capturing the lower-rank
    structure of the user-item interactions.

    Args:
        - n_users (int): Number of unique users.
        - n_items (int): Number of unique items.
        - emb_dim (int): Embedding dimension for user/item embeddings.

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
        emb_dim: int = 128,
        **kwargs,
    ):
        super().__init__()
        _ = kwargs
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        u = batch["user_id"].long()
        i = batch["item_id"].long()

        p_u = self.user_embedding(u)
        q_i = self.item_embedding(i)

        u_b = self.user_bias(u).squeeze(1)
        i_b = self.item_bias(i).squeeze(1)

        out = torch.sum(p_u * q_i, dim=1) + u_b + i_b

        return out


class NeuralMF(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 128,
        hidden_dims: list[int] = [256, 128, 64, 32, 16],
        **kwargs,
    ):
        super().__init__()
        _ = kwargs
        # Embeddings for GMF
        self.P = nn.Embedding(n_users, emb_dim)
        self.Q = nn.Embedding(n_items, emb_dim)

        # Embeddings for MLP
        self.U = nn.Embedding(n_users, emb_dim)
        self.V = nn.Embedding(n_items, emb_dim)

        layers = []
        input_size = emb_dim * 2

        for h in hidden_dims:
            layers += [
                nn.Linear(input_size, h),
                nn.ReLU(),
            ]
            input_size = h

        self.mlp = nn.Sequential(*layers)

        final_input_size = emb_dim + hidden_dims[-1]
        self.pred_layer = nn.Linear(final_input_size, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        u = batch["user_id"].long()
        i = batch["item_id"].long()

        # GMF part
        p_mf = self.P(u)  # [batch_size, emb_dim]
        q_mf = self.Q(i)  # [batch_size, emb_dim]
        gmf = p_mf * q_mf  # [batch_size, emb_dim]

        # MLP part
        p_mlp = self.U(u)  # [batch_size, emb_dim]
        q_mlp = self.V(i)  # [batch_size, emb_dim]
        mlp_input = torch.cat([p_mlp, q_mlp], dim=1)  # [batch_size, emb_dim*2]
        mlp_output = self.mlp(mlp_input)  # [batch_size, nums_hiddens[-1]]

        # Concatenate GMF and MLP outputs
        concat = torch.cat([gmf, mlp_output], dim=1)

        prediction = self.pred_layer(concat)

        return prediction.squeeze(1)
