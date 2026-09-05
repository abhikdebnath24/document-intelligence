from __future__ import annotations

from pathlib import Path

from docintel.config import load_config
from docintel.core.types import Chunk, RetrievalQuery, RetrievedChunk
from docintel.ingestion.loaders import agreement_type_for, is_upload_path
from docintel.retrieval.fusion import RRFFusion
from docintel.retrieval.pipeline import RetrievalPipeline, _with_unknown_type
from docintel.retrieval.rerankers import NoOpReranker
from docintel.retrieval.transforms import FilterExtractor

ROOT = Path(__file__).resolve().parents[2]


def _row(cid: str, doc_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=cid, doc_id=doc_id, text=cid, page_start=1, page_end=1, chunk_idx=0),
        score=1.0,
        source="dense",
        rank=1,
    )


class _FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        filt = dict(query.filters or {})
        self.calls.append(filt)
        if filt.get("doc_id") == "securian-up":
            return [_row("upload-hit", "securian-up")]
        return [_row("corpus-hit", "other")]


def test_is_upload_path() -> None:
    assert is_upload_path("data/uploads/abc.pdf")
    assert is_upload_path(r"C:\repo\data\uploads\abc.pdf")
    assert not is_upload_path("data/CUAD_v1/full_contract_pdf/Part_III/Maintenance/x.PDF")


def test_agreement_type_from_upload_filename(tmp_path: Path) -> None:
    path = (
        tmp_path
        / "uploads"
        / "SECURIAN-NET INVESTMENT INCOME MAINTENANCE AGREEMENT__deadbeef.pdf"
    )
    path.parent.mkdir()
    path.write_bytes(b"%PDF")
    assert agreement_type_for(path) == "Maintenance"


def test_with_unknown_type_keeps_uploads_visible() -> None:
    assert _with_unknown_type({"agreement_type": "Maintenance"})["agreement_type"] == [
        "Maintenance",
        "Unknown",
    ]
    assert _with_unknown_type({}) == {}
    assert _with_unknown_type({"agreement_type": "Unknown"}) == {"agreement_type": "Unknown"}


def test_search_widens_type_filter_with_one_retrieve() -> None:
    # One retrieve per query text. A second per-upload retrieve doubled the
    # grader input and pushed rewrite-heavy questions past query_deadline_s.
    cfg = load_config("dev_cpu", repo_root=ROOT)
    fake = _FakeRetriever()
    pipe = RetrievalPipeline(cfg, fake, RRFFusion(), NoOpReranker(), [], FilterExtractor([]))
    pipe.search(
        RetrievalQuery(
            text="what must Advantus do under the net investment income maintenance agreement?",
            k=5,
        )
    )
    assert len(fake.calls) == 1
    assert fake.calls[0] == {"agreement_type": ["Maintenance", "Unknown"]}


def test_search_rereads_catalog_for_upload_doc_hint() -> None:
    cfg = load_config("dev_cpu", repo_root=ROOT)
    fake = _FakeRetriever()
    rows: list[dict[str, str]] = []
    pipe = RetrievalPipeline(
        cfg,
        fake,
        RRFFusion(),
        NoOpReranker(),
        [],
        FilterExtractor([]),
        catalog_fn=lambda: list(rows),
    )
    q = RetrievalQuery(text="what must securianfundstrust reimburse", k=5)
    pipe.search(q)
    assert "doc_id" not in fake.calls[0]
    rows.append(
        {
            "doc_id": "securian-up",
            "doc_stem": "securianfundstrust_05_01_2012-maintenance agreement__deadbeef",
            "agreement_type": "Unknown",
        }
    )
    fake.calls.clear()
    hits = pipe.search(q)
    assert fake.calls[0]["doc_id"] == "securian-up"
    assert {r.chunk.chunk_id for r in hits} == {"upload-hit"}
