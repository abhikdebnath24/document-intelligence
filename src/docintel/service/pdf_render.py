from __future__ import annotations

from pathlib import Path

from docintel.core.types import BBox, Citation


def resolve_pdf_path(source_path: str, repo_root: Path) -> Path | None:
    raw = Path(source_path)
    for candidate in (raw, repo_root / raw, Path.cwd() / raw):
        if candidate.is_file():
            return candidate
    return None


def render_cited_page(pdf_path: Path, citation: Citation, *, dpi: int = 120) -> bytes:
    import pymupdf

    pdf = pymupdf.open(pdf_path)  # type: ignore[no-untyped-call]
    try:
        idx = max(citation.page_no, 1) - 1
        if idx >= pdf.page_count:
            idx = 0
        page = pdf.load_page(idx)  # type: ignore[no-untyped-call]
        boxes = citation.bboxes or []
        for box in boxes:
            _highlight(page, box)
        png: bytes = page.get_pixmap(dpi=dpi).tobytes("png")
        return png
    finally:
        pdf.close()  # type: ignore[no-untyped-call]


def _highlight(page: object, box: BBox) -> None:
    import pymupdf

    rect = pymupdf.Rect(box.x0, box.y0, box.x1, box.y1)  # type: ignore[no-untyped-call]
    if rect.is_empty or rect.is_infinite:
        return
    page.draw_rect(  # type: ignore[attr-defined]
        rect,
        color=(0.85, 0.65, 0.1),
        fill=(1.0, 0.92, 0.35),
        fill_opacity=0.35,
        width=1.0,
    )
