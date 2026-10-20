from __future__ import annotations

import re
from collections.abc import Iterable

from app.domain import ChunkDraft, PageText

_TOKEN = re.compile(r"\S+")
_SENTENCE_END = re.compile(r"[.!?][\"'”’')\]]*$")


def normalize_page_text(text: str) -> str:
    """Normalize layout noise without flattening useful paragraph boundaries."""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\f\v ]+", " ", text)
    text = re.sub(r"\n[ ]+\n", "\n\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class Chunker:
    """Page-aware, overlap-preserving splitter with paragraph/sentence preferences.

    Token counts are a conservative whitespace-token approximation, deliberately
    avoiding a tokenizer download at ingestion time. A chunk never crosses a PDF
    page boundary, which keeps page citations precise.
    """

    def __init__(self, max_tokens: int = 420, overlap_tokens: int = 64) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if overlap_tokens < 0 or overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be non-negative and smaller than max_tokens")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def split_pages(self, pages: Iterable[PageText]) -> list[ChunkDraft]:
        result: list[ChunkDraft] = []
        for page in pages:
            text = normalize_page_text(page.text)
            if not text:
                continue
            result.extend(self._split_page(text, page.page_number, len(result)))
        return result

    def _split_page(self, text: str, page_number: int | None, first_index: int) -> list[ChunkDraft]:
        matches = list(_TOKEN.finditer(text))
        if not matches:
            return []

        drafts: list[ChunkDraft] = []
        start = 0
        while start < len(matches):
            end = min(start + self.max_tokens, len(matches))
            if end < len(matches):
                end = self._preferred_end(text, matches, start, end)
            if end <= start:
                end = min(start + self.max_tokens, len(matches))

            piece = text[matches[start].start() : matches[end - 1].end()].strip()
            if piece:
                drafts.append(
                    ChunkDraft(
                        chunk_index=first_index + len(drafts),
                        page_number=page_number,
                        text=piece,
                        token_count=end - start,
                    )
                )
            if end >= len(matches):
                break
            start = max(start + 1, end - self.overlap_tokens)
        return drafts

    def _preferred_end(
        self,
        text: str,
        matches: list[re.Match[str]],
        start: int,
        desired_end: int,
    ) -> int:
        """Move a cut to the latest nearby paragraph or sentence boundary."""
        lower_bound = min(desired_end, start + max(1, int(self.max_tokens * 0.62)))
        paragraph_end: int | None = None
        sentence_end: int | None = None

        for boundary in range(lower_bound, desired_end + 1):
            if boundary <= 0 or boundary >= len(matches):
                continue
            previous = matches[boundary - 1].group()
            gap = text[matches[boundary - 1].end() : matches[boundary].start()]
            if "\n\n" in gap:
                paragraph_end = boundary
            elif gap and _SENTENCE_END.search(previous):
                sentence_end = boundary

        return paragraph_end or sentence_end or desired_end
