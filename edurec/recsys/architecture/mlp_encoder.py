from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class MLPEncoderConfig:
    num_dense_features: int
    categorical_cardinalities: list[int] = field(default_factory=list)
    output_dim: int = 128
    hidden_dims: list[int] = field(default_factory=list)
    dropout: float = 0.1
    num_text_features: int = 0
    use_text_features: bool = True


class MLPEncoder(nn.Module):
    def __init__(self, cfg: MLPEncoderConfig):
        super().__init__()
        self.num_dense_features = cfg.num_dense_features
        self.output_dim = cfg.output_dim

        if not 0 <= cfg.num_text_features <= cfg.num_dense_features:
            raise ValueError(
                "num_text_features must be in [0, num_dense_features], got "
                f"{cfg.num_text_features} for {cfg.num_dense_features} dense features."
            )

        # Text embeddings are stored as the trailing block of the dense features
        # (dense, then text, then categorical), so dropping text means slicing
        # the dense block and keeping the categorical block untouched.
        self.num_input_dense = (
            cfg.num_dense_features
            if cfg.use_text_features
            else cfg.num_dense_features - cfg.num_text_features
        )

        cat_emb_dim = max(4, min(cfg.output_dim // 2, 32))
        self.cat_embeddings = nn.ModuleList(
            nn.Embedding(cardinality + 1, cat_emb_dim, padding_idx=0)
            for cardinality in cfg.categorical_cardinalities
        )

        input_dim = self.num_input_dense + len(self.cat_embeddings) * cat_emb_dim
        hidden_dims = cfg.hidden_dims or [cfg.output_dim * 2]

        if input_dim == 0:
            self.network = None
        else:
            layers: list[nn.Module] = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(prev_dim, hidden_dim),
                        nn.GELU(),
                        nn.Dropout(cfg.dropout),
                    ]
                )
                prev_dim = hidden_dim

            layers.extend(
                [
                    nn.Linear(prev_dim, cfg.output_dim),
                    nn.LayerNorm(cfg.output_dim),
                ]
            )
            self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.network is None:
            return features.new_zeros((*features.shape[:-1], self.output_dim))

        encoded_parts: list[torch.Tensor] = []

        if self.num_input_dense > 0:
            encoded_parts.append(features[..., : self.num_input_dense].float())

        if self.cat_embeddings:
            cat_ids = features[..., self.num_dense_features :].long() + 1
            for idx, embedding in enumerate(self.cat_embeddings):
                encoded_parts.append(embedding(cat_ids[..., idx]))

        stacked = torch.cat(encoded_parts, dim=-1)
        return self.network(stacked)
