from __future__ import annotations

from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path

from app.domain import PageText
from app.services.chunking import normalize_page_text

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}


class DocumentParseError(ValueError):
    """The uploaded file is supported but could not be read as a document."""


class _ReadableHTML(HTMLParser):
    _BLOCK_TAGS = {
        "address",
        "article",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hr",
        "li",
        "main",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
    _IGNORED_TAGS = {"script", "style", "svg", "noscript"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._IGNORED_TAGS:
            self._ignored_depth += 1
        elif not self._ignored_depth and tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._IGNORED_TAGS and self._ignored_depth:
            self._ignored_depth -= 1
        elif not self._ignored_depth and tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)


def parse_document(filename: str, content: bytes) -> list[PageText]:
    """Extract readable text from a supported upload without writing it to disk."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DocumentParseError(f"Unsupported file type. Supported extensions: {allowed}")
    if not content:
        raise DocumentParseError("The uploaded file is empty.")

    try:
        if suffix == ".pdf":
            pages = _parse_pdf(content)
        elif suffix == ".docx":
            pages = _parse_docx(content)
        elif suffix in {".html", ".htm"}:
            parser = _ReadableHTML()
            parser.feed(_decode_text(content))
            parser.close()
            pages = [PageText(page_number=None, text="".join(parser.parts))]
        else:
            pages = [PageText(page_number=None, text=_decode_text(content))]
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError(
            "The file could not be parsed. Check that it is a valid, unencrypted document."
        ) from exc

    cleaned = [PageText(page.page_number, normalize_page_text(page.text)) for page in pages]
    if not any(page.text for page in cleaned):
        raise DocumentParseError(
            "No selectable text was found. Scanned PDFs need OCR before they can be indexed."
        )
    return cleaned


def _decode_text(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("cp1252")


def _parse_pdf(content: bytes) -> list[PageText]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency is declared in requirements
        raise DocumentParseError("PDF support is unavailable; install the pypdf dependency.") from exc

    reader = PdfReader(BytesIO(content), strict=False)
    pages: list[PageText] = []
    for index, page in enumerate(reader.pages, start=1):
        pages.append(PageText(page_number=index, text=page.extract_text() or ""))
    return pages


def _parse_docx(content: bytes) -> list[PageText]:
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover - dependency is declared in requirements
        raise DocumentParseError("DOCX support is unavailable; install the python-docx dependency.") from exc

    document = Document(BytesIO(content))
    parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return [PageText(page_number=None, text="\n\n".join(parts))]
