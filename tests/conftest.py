import numpy as np
import pytest

from edurec import config
from edurec.datasets import data_processor as data_processor_module


class FakeSentenceTransformer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.max_seq_length = config.TEXT_MAX_TOKENS
        self.last_texts: list[str] = []

    def get_sentence_embedding_dimension(self) -> int:
        return config.TEXT_EMBEDDING_DIM

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        _ = batch_size, show_progress_bar, convert_to_numpy
        self.last_texts = list(texts)

        embeddings = np.zeros(
            (len(texts), config.TEXT_EMBEDDING_DIM),
            dtype=np.float32,
        )
        for row_idx, text in enumerate(texts):
            for char_idx, byte in enumerate(text.encode("utf-8")):
                embeddings[row_idx, char_idx % config.TEXT_EMBEDDING_DIM] += (
                    byte % 17
                ) + 1
        return embeddings


@pytest.fixture
def fake_sentence_model(monkeypatch: pytest.MonkeyPatch):
    models: dict[str, FakeSentenceTransformer] = {}

    def fake_loader(model_name: str) -> FakeSentenceTransformer:
        model = models.get(model_name)
        if model is None:
            model = FakeSentenceTransformer(model_name)
            models[model_name] = model
        return model

    monkeypatch.setattr(
        data_processor_module, "_get_sentence_embedding_model", fake_loader
    )
    return models


@pytest.fixture(autouse=True)
def _auto_fake_sentence_model(fake_sentence_model):
    return fake_sentence_model
