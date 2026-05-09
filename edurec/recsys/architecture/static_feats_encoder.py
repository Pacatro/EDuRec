import torch
from torch import nn


class FeatureInteractionEncoder(nn.Module):
    def __init__(
        self,
        num_dense_features: int,
        categorical_cardinalities: list[int],
        emb_dim: int,
    ):
        super().__init__()
        # Mantenemos tu lógica original para proyectar cada feature
        self.num_dense_features = num_dense_features
        self.emb_dim = emb_dim

        self.dense_proj = (
            nn.Linear(num_dense_features, emb_dim) if num_dense_features > 0 else None
        )
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(c + 1, emb_dim, padding_idx=0)
                for c in categorical_cardinalities
            ]
        )

        # --- NUEVO: Bloque de Interacción ---
        # Calculamos cuántos vectores vamos a tener (ID + densas + cada categórica)
        self.total_components = (
            1 + (1 if num_dense_features > 0 else 0) + len(categorical_cardinalities)
        )

        self.fusion = nn.Sequential(
            nn.Linear(self.total_components * emb_dim, emb_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, base_id_emb, x_static):
        # 1. Recolectamos todos los "conceptos" en una lista
        features = [base_id_emb]  # El ID es nuestra base

        if self.dense_proj is not None:
            dense = x_static[..., : self.num_dense_features].float()
            features.append(self.dense_proj(dense))

        if self.cat_embeddings:
            cat_ids = x_static[..., self.num_dense_features :].long() + 1
            for i, emb_layer in enumerate(self.cat_embeddings):
                features.append(emb_layer(cat_ids[..., i]))

        # 2. Interacción: Concatenamos y proyectamos
        # Esto permite que el modelo aprenda pesos específicos para cada combinación
        combined = torch.cat(features, dim=-1)
        interacted = self.fusion(combined)

        # 3. Conexión residual para estabilidad
        return base_id_emb + interacted
