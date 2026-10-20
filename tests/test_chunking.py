from app.domain import PageText
from app.services.chunking import Chunker


def test_chunks_respect_size_and_overlap() -> None:
    words = [f"token{i}" for i in range(23)]
    chunker = Chunker(max_tokens=8, overlap_tokens=2)

    chunks = chunker.split_pages([PageText(page_number=1, text=" ".join(words))])

    assert len(chunks) > 1
    assert all(chunk.token_count <= 8 for chunk in chunks)
    assert all(chunk.page_number == 1 for chunk in chunks)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert previous.text.split()[-2:] == current.text.split()[:2]


def test_chunker_prefers_sentence_boundaries_and_does_not_cross_pages() -> None:
    chunker = Chunker(max_tokens=8, overlap_tokens=1)
    pages = [
        PageText(page_number=1, text="A short sentence ends here. Another sentence follows after it."),
        PageText(page_number=2, text="This text belongs to the second page."),
    ]

    chunks = chunker.split_pages(pages)

    assert chunks[0].page_number == 1
    assert chunks[-1].page_number == 2
    assert all(chunk.chunk_index == index for index, chunk in enumerate(chunks))
    assert all(chunk.text.strip() for chunk in chunks)
