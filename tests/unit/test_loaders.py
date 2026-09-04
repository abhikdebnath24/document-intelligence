from __future__ import annotations

from docintel.core.types import BBox, Page, TextBlock
from docintel.ingestion.chunkers import join_pages
from docintel.ingestion.loaders import PAGE_NUM_RE, _strip_headers_footers, pdf_txt_ratio


def _block(text: str) -> TextBlock:
    return TextBlock(text=text, bbox=BBox(x0=1, y0=1, x1=9, y1=9))


def test_header_footer_and_page_number_stripped_from_lines() -> None:
    pages = [
        Page(page_no=i, text=f"ACME CORP\nbody {i} unique text\nPage {i} of 4", blocks=[])
        for i in range(1, 5)
    ]
    cleaned = _strip_headers_footers(pages)
    joined = "\n".join(p.text for p in cleaned)
    assert "ACME CORP" not in joined
    assert "Page 1 of 4" not in joined
    assert "body 1 unique text" in joined


def test_header_stripping_reaches_blocks_seen_by_chunkers() -> None:
    pages = [
        Page(
            page_no=i,
            text="",
            blocks=[_block("ACME CORP"), _block(f"body {i} unique text"), _block(f"- {i} -")],
        )
        for i in range(1, 5)
    ]
    cleaned = _strip_headers_footers(pages)
    full, spans = join_pages(cleaned)
    assert "ACME CORP" not in full
    assert "- 2 -" not in full
    assert "body 3 unique text" in full
    assert len(spans) == 4
    assert all(p.text == "\n".join(b.text for b in p.blocks) for p in cleaned)


def test_page_number_regex_keeps_years() -> None:
    assert PAGE_NUM_RE.match("12")
    assert PAGE_NUM_RE.match("Page 3 of 40")
    assert PAGE_NUM_RE.match("- 7 -")
    assert not PAGE_NUM_RE.match("2010")
    assert not PAGE_NUM_RE.match("1.")


def test_pdf_txt_ratio_gate() -> None:
    ratio, ok = pdf_txt_ratio(100, 100)
    assert ratio == 1.0 and ok
    ratio, ok = pdf_txt_ratio(4, 200)
    assert ok is False
    assert pdf_txt_ratio(0, 10) == (None, False)
