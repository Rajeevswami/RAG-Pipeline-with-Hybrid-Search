import json

import pytest

from app.services.evaluation import metrics_at_k
from scripts.evaluate_retrieval import load_dataset


def test_retrieval_metrics_at_k() -> None:
    metrics = metrics_at_k(["irrelevant", "hit-1", "hit-2"], {"hit-1", "hit-2"}, k=2)

    assert metrics.recall_at_k == 0.5
    assert metrics.precision_at_k == 0.5
    assert metrics.reciprocal_rank == 0.5
    assert 0 < metrics.ndcg_at_k < 1


def test_evaluation_dataset_accepts_document_level_goldens(tmp_path) -> None:
    path = tmp_path / "goldens.jsonl"
    path.write_text(
        json.dumps({"question": "What is the leave policy?", "relevant_document_ids": ["doc-1"]}) + "\n",
        encoding="utf-8",
    )

    samples = load_dataset(path)

    assert len(samples) == 1
    assert samples[0].relevance_level == "document"
    assert samples[0].relevant_ids == {"doc-1"}


def test_evaluation_dataset_requires_one_relevance_field(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"question": "Policy?"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one"):
        load_dataset(path)
