from torch import nn
import torch


class StaticFeatureEncoder(nn.Module):
    def __init__(
        self,
        num_dense_features: int,
        categorical_cardinalities: list[int],
        emb_dim: int,
    ):
        super().__init__()
        self.num_dense_features = num_dense_features
        self.num_categorical_features = len(categorical_cardinalities)
        self.emb_dim = emb_dim

        self.dense_proj = (
            nn.Linear(num_dense_features, emb_dim, bias=False)
            if num_dense_features > 0
            else None
        )
        self.cat_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, emb_dim, padding_idx=0)
            for cardinality in categorical_cardinalities
        )
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.new_zeros((*x.shape[:-1], self.emb_dim))

        if self.dense_proj is not None:
            dense_feats = x[..., : self.num_dense_features].float()
            out = out + self.dense_proj(dense_feats)

        if self.cat_embeddings:
            cat_feats = x[..., self.num_dense_features :].long() + 1
            cat_feats = cat_feats.clamp(min=0)
            for idx, embedding in enumerate(self.cat_embeddings):
                out = out + embedding(cat_feats[..., idx])

        return self.norm(out)
