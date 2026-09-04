from __future__ import annotations

from docintel.core.interfaces import BaseChunker
from docintel.core.types import BBox, Chunk, Document, Page

CHARS_PER_TOKEN = 4.0
_SEPARATORS = ("\n\n", "\n", " ")


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN)) if text else 0


def join_pages(pages: list[Page]) -> tuple[str, list[tuple[int, int, int, BBox]]]:
    """Return full text and (start, end, page_no, bbox) ranges for each block."""
    parts: list[str] = []
    spans: list[tuple[int, int, int, BBox]] = []
    cursor = 0
    for i, page in enumerate(pages):
        if i:
            parts.append("\n")
            cursor += 1
        if page.blocks:
            for j, block in enumerate(page.blocks):
                if j:
                    parts.append("\n")
                    cursor += 1
                start = cursor
                parts.append(block.text)
                cursor += len(block.text)
                spans.append((start, cursor, page.page_no, block.bbox))
        else:
            start = cursor
            parts.append(page.text)
            cursor += len(page.text)
            if page.text:
                spans.append((start, cursor, page.page_no, BBox(x0=0, y0=0, x1=0, y1=0)))
    return "".join(parts), spans


def spans_for_range(
    start: int, end: int, block_spans: list[tuple[int, int, int, BBox]]
) -> tuple[int, int, list[BBox]]:
    pages: list[int] = []
    boxes: list[BBox] = []
    for b_start, b_end, page_no, bbox in block_spans:
        if b_end <= start or b_start >= end:
            continue
        pages.append(page_no)
        if bbox.x1 > bbox.x0 or bbox.y1 > bbox.y0:
            boxes.append(bbox)
    if not pages:
        return 1, 1, []
    return min(pages), max(pages), boxes


def _split_keep(text: str, sep: str) -> list[str]:
    if not sep:
        return list(text)
    parts = text.split(sep)
    if len(parts) == 1:
        return parts
    out: list[str] = []
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            out.append(part + sep)
        elif part:
            out.append(part)
    return out


def _window(pieces: list[tuple[int, str]], size: int, overlap: int) -> list[tuple[int, int, str]]:
    if not pieces:
        return []
    chunks: list[tuple[int, int, str]] = []
    start_i = 0
    n = len(pieces)
    while start_i < n:
        buf: list[str] = []
        end_i = start_i
        start_char = pieces[start_i][0]
        used = 0
        while end_i < n:
            piece = pieces[end_i][1]
            if buf and used + len(piece) > size:
                break
            buf.append(piece)
            used += len(piece)
            end_i += 1
            if used >= size:
                break
        text = "".join(buf)
        end_char = start_char + len(text)
        if text.strip():
            chunks.append((start_char, end_char, text))
        if end_i >= n:
            break
        # step forward, keep overlap chars
        keep = 0
        next_i = end_i
        while next_i > start_i and keep < overlap:
            next_i -= 1
            keep += len(pieces[next_i][1])
        start_i = max(start_i + 1, next_i)
    return chunks


def _pieces_from_text(text: str) -> list[tuple[int, str]]:
    pieces: list[tuple[int, str]] = [(0, text)]
    for sep in _SEPARATORS:
        next_pieces: list[tuple[int, str]] = []
        for start, blob in pieces:
            if len(blob) <= 1:
                next_pieces.append((start, blob))
                continue
            offset = start
            for part in _split_keep(blob, sep):
                next_pieces.append((offset, part))
                offset += len(part)
        pieces = next_pieces
    return pieces


def chunk_windows(
    doc: Document,
    *,
    chunk_tokens: int,
    overlap_tokens: int,
    contextual_header: bool,
    alias: str,
) -> list[Chunk]:
    text, block_spans = join_pages(doc.pages)
    size = max(1, int(chunk_tokens * CHARS_PER_TOKEN))
    overlap = max(0, int(overlap_tokens * CHARS_PER_TOKEN))
    pieces = _pieces_from_text(text) if text else []
    windows = _window(pieces, size, overlap)
    chunks: list[Chunk] = []
    header = doc.agreement_type if contextual_header else None
    for idx, (start, end, raw) in enumerate(windows):
        page_start, page_end, bboxes = spans_for_range(start, end, block_spans)
        body = raw
        if header:
            body = f"{header}\n{raw}"
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}:{idx}",
                doc_id=doc.doc_id,
                text=body,
                page_start=page_start,
                page_end=page_end,
                bboxes=bboxes,
                section_header=header,
                chunk_idx=idx,
                char_span=(start, end),
                metadata={"chunker": alias},
            )
        )
    return chunks


class RecursiveChunker(BaseChunker):
    def __init__(
        self,
        chunk_tokens: int = 512,
        overlap_tokens: int = 64,
        contextual_header: bool = False,
        **_: object,
    ) -> None:
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.contextual_header = contextual_header

    def alias(self) -> str:
        return "recursive"

    def chunk(self, doc: Document) -> list[Chunk]:
        return chunk_windows(
            doc,
            chunk_tokens=self.chunk_tokens,
            overlap_tokens=self.overlap_tokens,
            contextual_header=self.contextual_header,
            alias=self.alias(),
        )


class FixedTokenChunker(BaseChunker):
    def __init__(
        self,
        chunk_tokens: int = 512,
        overlap_tokens: int = 64,
        contextual_header: bool = False,
        **_: object,
    ) -> None:
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.contextual_header = contextual_header

    def alias(self) -> str:
        return "fixed_token"

    def chunk(self, doc: Document) -> list[Chunk]:
        text, block_spans = join_pages(doc.pages)
        size = max(1, int(self.chunk_tokens * CHARS_PER_TOKEN))
        step = max(1, size - int(self.overlap_tokens * CHARS_PER_TOKEN))
        chunks: list[Chunk] = []
        header = doc.agreement_type if self.contextual_header else None
        idx = 0
        pos = 0
        n = len(text)
        if n == 0:
            return []
        while pos < n:
            end = min(n, pos + size)
            raw = text[pos:end]
            if raw.strip():
                page_start, page_end, bboxes = spans_for_range(pos, end, block_spans)
                body = f"{header}\n{raw}" if header else raw
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}:{idx}",
                        doc_id=doc.doc_id,
                        text=body,
                        page_start=page_start,
                        page_end=page_end,
                        bboxes=bboxes,
                        section_header=header,
                        chunk_idx=idx,
                        char_span=(pos, end),
                        metadata={"chunker": self.alias()},
                    )
                )
                idx += 1
            if end >= n:
                break
            pos += step
        return chunks
