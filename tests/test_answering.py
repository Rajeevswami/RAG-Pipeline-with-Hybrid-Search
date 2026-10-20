import json

import pytest

from app.config import Settings
from app.domain import ChunkRecord, RetrievedChunk
from app.services.answering import AnswerService


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    async def complete(self, question, evidence):
        return self.response


def _retrieved(text: str) -> list[RetrievedChunk]:
    chunk = ChunkRecord(
        id="chunk-1",
        document_id="doc-1",
        filename="employee-handbook.pdf",
        chunk_index=2,
        page_number=4,
        text=text,
        token_count=len(text.split()),
    )
    return [RetrievedChunk(chunk=chunk, fusion_score=0.02, rerank_score=0.9)]


@pytest.mark.asyncio
async def test_verifies_source_id_and_exact_support_quote() -> None:
    excerpt = "Employees receive 20 days of paid time off each calendar year."
    response = json.dumps(
        {
            "answer": "Employees receive 20 days of paid time off each calendar year [S1].",
            "citations": [{"source_id": "S1", "quote": excerpt}],
            "abstained": False,
        }
    )
    service = AnswerService(Settings(_env_file=None), FakeLLM(response))

    result = await service.answer("How much paid time off do employees receive?", _retrieved(excerpt))

    assert result.citation_verified is True
    assert result.verification_method == "quote_match"
    assert result.citations[0].filename == "employee-handbook.pdf"
    assert result.citations[0].page_number == 4
    assert result.citations[0].chunk_id == "chunk-1"
    assert result.sources[0].chunk_id == "chunk-1"


@pytest.mark.asyncio
async def test_invalid_model_citation_uses_verbatim_fallback() -> None:
    excerpt = "Employees receive 20 days of paid time off each calendar year."
    response = json.dumps(
        {
            "answer": "Everyone receives unlimited leave [S9].",
            "citations": [{"source_id": "S9", "quote": "unlimited leave"}],
            "abstained": False,
        }
    )
    service = AnswerService(Settings(_env_file=None), FakeLLM(response))

    result = await service.answer("How much paid time off is there?", _retrieved(excerpt))

    assert result.citation_verified is True
    assert result.verification_method == "extractive_fallback"
    assert result.citations[0].source_id == "S1"
    assert "[S1]" in result.answer
    assert result.citations[0].quote in excerpt


@pytest.mark.asyncio
async def test_invalid_zero_source_id_is_rejected() -> None:
    excerpt = "Employees receive 20 days of paid time off each calendar year."
    response = json.dumps(
        {
            "answer": "Employees receive paid time off [S1] [S0].",
            "citations": [
                {"source_id": "S1", "quote": excerpt},
                {"source_id": "S0", "quote": excerpt},
            ],
            "abstained": False,
        }
    )
    service = AnswerService(Settings(_env_file=None), FakeLLM(response))

    result = await service.answer("How much paid time off is there?", _retrieved(excerpt))

    assert result.verification_method == "extractive_fallback"
    assert "[S0]" not in result.answer


@pytest.mark.asyncio
async def test_abstains_when_no_relevant_verbatim_sentence_exists() -> None:
    response = "not valid JSON"
    service = AnswerService(Settings(_env_file=None), FakeLLM(response))

    result = await service.answer(
        "What is the office paint color?", _retrieved("Submit receipts through the expense portal.")
    )

    assert result.abstained is True
    assert result.citation_verified is True
    assert result.citations == []
    assert result.verification_method == "abstained"
