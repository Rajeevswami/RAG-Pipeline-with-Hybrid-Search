from app.config import Settings
from app.domain import ChunkRecord
from app.services.reranking import Reranker
from app.services.retrieval import HybridRetriever


class FakeEmbeddings:
    def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeVectors:
    def search(self, vector, limit):
        return [("dense-only", 0.91), ("shared", 0.82)]


class FakeLexical:
    def search_lexical(self, query, limit):
        return ["sparse-only", "shared"]

    def get_chunks_by_ids(self, chunk_ids):
        texts = {
            "dense-only": "A general note about workplace guidelines.",
            "sparse-only": "The exact travel policy covers train bookings.",
            "shared": "The travel policy covers train bookings and approvals.",
        }
        return {
            chunk_id: ChunkRecord(chunk_id, "doc", "policy.txt", index, None, texts[chunk_id], 8)
            for index, chunk_id in enumerate(texts)
            if chunk_id in chunk_ids
        }


class KeepOrder:
    def rerank(self, query, candidates):
        return candidates


def test_rrf_combines_dense_and_bm25_ranks() -> None:
    settings = Settings(
        _env_file=None,
        dense_weight=0.5,
        lexical_weight=0.5,
        retrieval_candidates=5,
        default_top_k=5,
        max_top_k=5,
    )
    retriever = HybridRetriever(settings, FakeEmbeddings(), FakeVectors(), FakeLexical(), KeepOrder())

    results = retriever.retrieve("travel policy", top_k=5)

    assert [item.chunk.id for item in results] == ["shared", "dense-only", "sparse-only"]
    assert results[0].dense_rank == 2
    assert results[0].lexical_rank == 2
    assert results[0].fusion_score > results[1].fusion_score


def test_default_reranker_prefers_query_term_overlap() -> None:
    settings = Settings(_env_file=None)
    retriever = HybridRetriever(settings, FakeEmbeddings(), FakeVectors(), FakeLexical(), Reranker())

    results = retriever.retrieve("travel bookings approval", top_k=3)

    assert results[0].chunk.id == "shared"
    assert results[0].rerank_score > results[-1].rerank_score


def test_zero_weight_disables_that_retrieval_channel() -> None:
    settings = Settings(
        _env_file=None,
        dense_weight=0,
        lexical_weight=1,
        retrieval_candidates=5,
        default_top_k=5,
        max_top_k=5,
    )
    retriever = HybridRetriever(settings, FakeEmbeddings(), FakeVectors(), FakeLexical(), KeepOrder())

    results = retriever.retrieve("travel policy", top_k=5)

    assert [item.chunk.id for item in results] == ["sparse-only", "shared"]
    assert all(item.dense_rank is None for item in results)
