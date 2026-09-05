from __future__ import annotations

from pathlib import Path

from frontend.streamlit_app.client import InProcessClient, run_pdf_ingest

from docintel.config import load_config
from docintel.core.types import QueryLog
from docintel.feedback.repository import SqlAlchemyFeedbackRepository
from docintel.ingestion.pipeline import IngestReport
from docintel.service.feedback_service import FeedbackService

ROOT = Path(__file__).resolve().parents[2]


def test_client_is_lazy_and_updates_one_feedback_row() -> None:
    config = load_config("dev_cpu", repo_root=ROOT)
    repo = SqlAlchemyFeedbackRepository("sqlite:///:memory:")
    feedback = FeedbackService(repo)
    client = InProcessClient(config, feedback, ROOT)
    assert client._query is None

    feedback.log_query(
        QueryLog(
            query_id="q1",
            question="Which law governs?",
            route="corpus_technical",
            config_hash="abc",
            profile="dev_cpu",
            answer="Delaware",
        )
    )
    client.rate("q1", 3, tags=["incomplete"])
    client.rate("q1", 5, tags=["good"], comment="fixed")

    rows = repo.list_feedback("q1")
    assert len(rows) == 1
    assert rows[0].rating == 5
    assert rows[0].tags == ["good"]
    assert rows[0].comment == "fixed"
    client.close()


def test_ingest_keeps_live_query_service(tmp_path: Path) -> None:
    config = load_config("dev_cpu", repo_root=ROOT)
    repo = SqlAlchemyFeedbackRepository("sqlite:///:memory:")
    client = InProcessClient(config, FeedbackService(repo), ROOT)
    client._query = object()

    class _Ing:
        def ingest_paths(self, paths: list[Path], *, only_changed: bool = True) -> IngestReport:
            _ = paths, only_changed
            return IngestReport(profile="dev_cpu", index_sig="sig", collection="col", indexed=1)

    client.ingest = _Ing()  # type: ignore[assignment]
    client.ingest_paths([tmp_path / "x.pdf"])
    assert client._query is not None
    client._query = None
    client.close()


def test_run_pdf_ingest_upserts_and_reports_chunks(tmp_path: Path) -> None:
    import pymupdf

    from docintel.ingestion.pipeline import IngestReport

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()  # type: ignore[no-untyped-call]
    doc.new_page()
    doc.save(src)
    doc.close()

    seen: list[Path] = []

    class _Fake:
        repo_root = tmp_path

        def ingest_paths(self, paths: list[Path]) -> IngestReport:
            seen.extend(paths)
            return IngestReport(
                profile="gpu_default",
                index_sig="sig",
                collection="col",
                indexed=1,
                chunks=4,
            )

    out = run_pdf_ingest(_Fake(), src.read_bytes())  # type: ignore[arg-type]
    assert out["chunks"] == 4
    assert out["indexed"] == 1
    assert out["collection"] == "col"
    assert seen and seen[0].parent == tmp_path / "data" / "uploads"
