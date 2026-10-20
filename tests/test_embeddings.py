import pytest

from app.config import Settings
from app.services.embeddings import EmbeddingClient, EmbeddingError


class FakeLocalModel:
    def __init__(self, vectors):
        self.vectors = vectors

    def get_sentence_embedding_dimension(self):
        return len(self.vectors[0])

    def encode(self, texts, **kwargs):
        return self.vectors


def test_local_embedding_vectors_are_dimension_and_finiteness_checked() -> None:
    settings = Settings(_env_file=None, embedding_provider="sentence-transformers")
    client = EmbeddingClient(settings)
    client._local_model = FakeLocalModel([[0.25, -0.5], [1.0, 0.0]])

    assert client.dimension == 2
    assert client.embed(["first", "second"]) == [[0.25, -0.5], [1.0, 0.0]]

    client._local_model = FakeLocalModel([[float("nan"), 1.0]])
    with pytest.raises(EmbeddingError, match="non-finite"):
        client.embed(["bad vector"])


def test_embedding_result_count_must_match_input_count() -> None:
    settings = Settings(_env_file=None, embedding_provider="sentence-transformers")
    client = EmbeddingClient(settings)
    client._local_model = FakeLocalModel([[0.25, -0.5]])

    with pytest.raises(EmbeddingError, match="incomplete result set"):
        client.embed(["first", "second"])
