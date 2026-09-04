from __future__ import annotations

from docintel.core.types import BBox, Document, Page, TextBlock
from docintel.ingestion.chunkers import (
    FixedTokenChunker,
    RecursiveChunker,
    join_pages,
)


def _doc(text_pages: list[str]) -> Document:
    pages: list[Page] = []
    for i, text in enumerate(text_pages, start=1):
        pages.append(
            Page(
                page_no=i,
                text=text,
                blocks=[
                    TextBlock(
                        text=text,
                        bbox=BBox(x0=10, y0=10, x1=200, y1=40),
                    )
                ],
            )
        )
    return Document(
        doc_id="demo",
        source_path="demo.pdf",
        agreement_type="Service",
        sha256="a" * 64,
        pages=pages,
    )


def test_recursive_covers_text_and_is_monotonic() -> None:
    body = "alpha " * 80 + "\n\n" + "bravo " * 80
    doc = _doc([body])
    full, _ = join_pages(doc.pages)
    chunks = RecursiveChunker(chunk_tokens=40, overlap_tokens=8).chunk(doc)
    assert chunks
    starts = [c.char_span[0] for c in chunks if c.char_span]
    assert starts == sorted(starts)
    covered = [False] * len(full)
    for chunk in chunks:
        assert chunk.char_span is not None
        start, end = chunk.char_span
        assert start < end
        for i in range(start, end):
            covered[i] = True
        assert chunk.bboxes
        assert chunk.page_start >= 1
    assert all(covered), "recursive chunker must cover every character"


def test_fixed_token_windows_and_bboxes() -> None:
    doc = _doc(["word " * 200, "tail " * 40])
    chunks = FixedTokenChunker(chunk_tokens=32, overlap_tokens=8).chunk(doc)
    assert len(chunks) >= 2
    spans = [c.char_span for c in chunks]
    assert all(span and span[0] < span[1] for span in spans)
    assert all(c.bboxes for c in chunks)
    assert chunks[-1].page_end >= chunks[0].page_start


def test_contextual_header_prepends_agreement_type() -> None:
    doc = _doc(["The term of this agreement is two years."])
    chunks = RecursiveChunker(chunk_tokens=32, contextual_header=True).chunk(doc)
    assert chunks
    assert chunks[0].text.startswith("Service\n")
    assert chunks[0].section_header == "Service"
