from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from docintel.core.interfaces import BaseLoader
from docintel.core.types import BBox, Document, Page, TextBlock
from docintel.data.corpus import agreement_type_from_folder, normalize_stem
from docintel.evaluation.gold import file_sha256

# "3", "- 3 -", "Page 3 of 12"; capped at 3 digits so a lone year is kept
PAGE_NUM_RE = re.compile(
    r"^-?\s*(?:page\s+)?\d{1,3}(?:\s*(?:of|/)\s*\d{1,3})?\s*-?$",
    re.IGNORECASE,
)
RATIO_LO = 0.97
RATIO_HI = 1.03
_ZERO_BBOX = BBox(x0=0, y0=0, x1=0, y1=0)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def doc_id_for(path: Path) -> str:
    return normalize_stem(path.stem)


def agreement_type_for(path: Path) -> str:
    try:
        return agreement_type_from_folder(path.parent.name)
    except ValueError:
        return "Unknown"


def _norm_ratio_text(text: str) -> str:
    return "".join(text.lower().split())


def norm_chars(text: str) -> int:
    return len(_norm_ratio_text(text))


def pdf_txt_ratio(pdf_norm_chars: int, txt_norm_chars: int) -> tuple[float | None, bool]:
    if pdf_norm_chars == 0 or txt_norm_chars == 0:
        return None, False
    ratio = pdf_norm_chars / txt_norm_chars
    return ratio, RATIO_LO <= ratio <= RATIO_HI


def _line_key(text: str) -> str:
    return " ".join(text.lower().split())


def _units(page: Page) -> list[TextBlock]:
    if page.blocks:
        return list(page.blocks)
    return [TextBlock(text=ln, bbox=_ZERO_BBOX) for ln in page.text.splitlines() if ln.strip()]


def _strip_headers_footers(pages: list[Page]) -> list[Page]:
    """Drop repeated first/last blocks and page-number-only blocks.

    Operates on blocks so chunkers (which read `page.blocks`) see the result.
    ponytail: block granularity; a header glued into a body block is not split out.
    """
    units = [_units(p) for p in pages]
    drop_first: set[str] = set()
    drop_last: set[str] = set()
    if len(pages) >= 3:
        thresh = max(2, (len(pages) + 1) // 2)
        first: dict[str, int] = {}
        last: dict[str, int] = {}
        for us in units:
            if not us:
                continue
            first[_line_key(us[0].text)] = first.get(_line_key(us[0].text), 0) + 1
            last[_line_key(us[-1].text)] = last.get(_line_key(us[-1].text), 0) + 1
        drop_first = {k for k, n in first.items() if n >= thresh}
        drop_last = {k for k, n in last.items() if n >= thresh}

    out: list[Page] = []
    for page, us in zip(pages, units, strict=True):
        keep: list[TextBlock] = []
        for i, unit in enumerate(us):
            key = _line_key(unit.text)
            if i == 0 and key in drop_first:
                continue
            if i == len(us) - 1 and key in drop_last:
                continue
            if PAGE_NUM_RE.match(unit.text.strip()):
                continue
            keep.append(unit)
        out.append(
            Page(
                page_no=page.page_no,
                text="\n".join(u.text for u in keep),
                blocks=keep if page.blocks else [],
            )
        )
    return out


class PyMuPDFLoader(BaseLoader):
    def __init__(self, strip_headers_footers: bool = True, **_: object) -> None:
        self.strip_headers_footers = strip_headers_footers

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def alias(self) -> str:
        return "pymupdf"

    def load(self, path: Path) -> Document:
        import pymupdf

        pdf: Any = pymupdf.open(path)  # type: ignore[no-untyped-call]
        try:
            pages: list[Page] = []
            for i in range(int(pdf.page_count)):
                page = pdf.load_page(i)
                data = page.get_text("dict")
                blocks: list[TextBlock] = []
                lines_out: list[str] = []
                for block in data.get("blocks", []):
                    if int(block.get("type", 1)) != 0:
                        continue
                    bbox = block.get("bbox") or (0.0, 0.0, 0.0, 0.0)
                    parts: list[str] = []
                    for line in block.get("lines", []):
                        # spans are contiguous runs; a space here would split words on font changes
                        span_text = "".join(
                            str(span.get("text", "")) for span in line.get("spans", [])
                        )
                        if span_text.strip():
                            parts.append(span_text)
                    text = "\n".join(parts).strip()
                    if not text:
                        continue
                    blocks.append(
                        TextBlock(
                            text=text,
                            bbox=BBox(
                                x0=float(bbox[0]),
                                y0=float(bbox[1]),
                                x1=float(bbox[2]),
                                y1=float(bbox[3]),
                            ),
                        )
                    )
                    lines_out.append(text)
                pages.append(Page(page_no=i + 1, text="\n".join(lines_out), blocks=blocks))
        finally:
            pdf.close()
        # ratio gate must see pre-strip text: TXT oracle still contains headers/page numbers
        raw_norm_chars = sum(norm_chars(p.text) for p in pages)
        if self.strip_headers_footers:
            pages = _strip_headers_footers(pages)
        return Document(
            doc_id=doc_id_for(path),
            source_path=str(path),
            agreement_type=agreement_type_for(path),
            sha256=file_sha256(path),
            pages=pages,
            metadata={"raw_norm_chars": raw_norm_chars},
        )


class TxtLoader(BaseLoader):
    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".txt"

    def alias(self) -> str:
        return "txt"

    def load(self, path: Path) -> Document:
        text = path.read_text(encoding="utf-8", errors="replace")
        page = Page(page_no=1, text=text, blocks=[])
        return Document(
            doc_id=doc_id_for(path),
            source_path=str(path),
            agreement_type=agreement_type_for(path),
            sha256=_sha256_bytes(text.encode("utf-8")),
            pages=[page],
        )


def validate_against_txt(doc: Document, txt_path: Path | None) -> dict[str, Any]:
    if txt_path is None or not txt_path.is_file():
        return {"doc_id": doc.doc_id, "ratio": None, "ok": None, "reason": "no_txt"}
    pdf_chars = int(
        doc.metadata.get("raw_norm_chars") or sum(norm_chars(p.text) for p in doc.pages)
    )
    txt_chars = norm_chars(txt_path.read_text(encoding="utf-8", errors="replace"))
    ratio, ok = pdf_txt_ratio(pdf_chars, txt_chars)
    return {"doc_id": doc.doc_id, "ratio": ratio, "ok": ok, "reason": None if ok else "ratio"}
