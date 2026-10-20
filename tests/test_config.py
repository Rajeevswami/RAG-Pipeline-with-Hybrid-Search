import pytest
from pydantic import ValidationError

from app.config import Settings


def test_provider_readiness_checks_the_selected_provider() -> None:
    openai = Settings(_env_file=None, openai_api_key="openai-key")
    anthropic = Settings(
        _env_file=None,
        llm_provider="anthropic",
        anthropic_api_key="anthropic-key",
        openai_api_key=None,
    )

    assert openai.has_llm_credentials is True
    assert openai.has_embedding_credentials is True
    assert anthropic.has_llm_credentials is True
    assert anthropic.has_embedding_credentials is False


def test_rejects_overlap_equal_to_chunk_size() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size_tokens=64, chunk_overlap_tokens=64)


def test_local_embedding_chunk_settings_respect_their_context_window() -> None:
    with pytest.raises(ValidationError, match="LOCAL_EMBEDDING_CHUNK_OVERLAP_TOKENS"):
        Settings(
            _env_file=None,
            embedding_provider="sentence-transformers",
            local_embedding_chunk_size_tokens=64,
            local_embedding_chunk_overlap_tokens=64,
        )
