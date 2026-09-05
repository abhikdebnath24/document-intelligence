from __future__ import annotations

from docintel.core.types import BBox, Citation
from docintel.service.pdf_render import highlight_boxes


def _cite(*, page_no: int, page_end: int, boxes: int) -> Citation:
    return Citation(
        chunk_id="c1",
        doc_id="d",
        page_no=page_no,
        page_end=page_end,
        bboxes=[BBox(x0=0, y0=0, x1=10, y1=10)] * boxes,
        quote="q",
    )


def test_highlight_boxes_skips_multi_page_chunk() -> None:
    assert highlight_boxes(_cite(page_no=2, page_end=4, boxes=2)) == []


def test_highlight_boxes_keeps_single_page() -> None:
    boxes = highlight_boxes(_cite(page_no=3, page_end=3, boxes=1))
    assert len(boxes) == 1
    assert highlight_boxes(_cite(page_no=3, page_end=0, boxes=1))
