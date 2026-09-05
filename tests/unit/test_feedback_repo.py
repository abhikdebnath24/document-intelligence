from __future__ import annotations

from pathlib import Path

import pytest

from docintel.core.types import DocumentRecord, QueryLog
from docintel.feedback.analytics import (
    export_csv,
    ratings_by_agreement_type,
    ratings_by_config_hash,
    ratings_by_route,
    worst_queries,
)
from docintel.feedback.repository import SqlAlchemyFeedbackRepository
from docintel.service.feedback_service import FeedbackService


def _repo() -> SqlAlchemyFeedbackRepository:
    return SqlAlchemyFeedbackRepository("sqlite:///:memory:")


def _log(**kwargs: object) -> QueryLog:
    payload: dict[str, object] = {
        "query_id": "q1",
        "question": "Which law governs?",
        "route": "corpus_technical",
        "config_hash": "abc123",
        "profile": "gpu_default",
        "answer": "New York",
        "cited_doc_ids": ["doc_a"],
    }
    payload.update(kwargs)
    return QueryLog.model_validate(payload)


def _doc(doc_id: str, agreement_type: str) -> DocumentRecord:
    return DocumentRecord(
        doc_id=doc_id,
        source_path=f"/tmp/{doc_id}.pdf",
        agreement_type=agreement_type,
        sha256="0" * 64,
        pipeline_version="v1",
        index_sig="sig",
        n_chunks=3,
        collection="cuad_test",
        status="indexed",
    )


def test_rate_roundtrip_and_unknown_query() -> None:
    repo = _repo()
    svc = FeedbackService(repo)
    svc.log_query(_log())
    item = svc.rate("q1", 1, tags=["good"], comment="ok")
    assert item.query_id == "q1"
    assert repo.list_feedback("q1")[0].rating == 1
    assert repo.get_query("q1") is not None
    with pytest.raises(ValueError, match="unknown query_id"):
        svc.rate("missing", -1)
    repo.close()


def test_rate_rejects_bad_rating_and_tag() -> None:
    repo = _repo()
    svc = FeedbackService(repo)
    svc.log_query(_log())
    with pytest.raises(ValueError, match="rating"):
        svc.rate("q1", 0)
    with pytest.raises(ValueError, match="unknown feedback tags"):
        svc.rate("q1", -1, tags=["nope"])
    repo.close()


def test_document_upsert_matches_ingest_columns() -> None:
    repo = _repo()
    rec = _doc("doc_a", "License")
    repo.upsert_document(rec)
    got = repo.get_document("doc_a")
    assert got is not None
    assert got.agreement_type == "License"
    assert got.index_sig == "sig"
    repo.close()


def test_shares_db_file_with_ingest_registry(tmp_path: Path, monkeypatch) -> None:
    from docintel.ingestion.registry_store import DocumentRegistry

    monkeypatch.chdir(tmp_path)
    url = "sqlite:///./docintel.db"
    registry = DocumentRegistry(url)
    registry.upsert(_doc("doc_a", "License"))
    registry.close()

    repo = SqlAlchemyFeedbackRepository(url)
    got = repo.get_document("doc_a")
    assert got is not None and got.agreement_type == "License"
    assert repo.db_url == "sqlite:///" + str((tmp_path / "docintel.db").resolve())
    repo.close()


def test_analytics_and_csv(tmp_path: Path) -> None:
    repo = _repo()
    svc = FeedbackService(repo)
    repo.upsert_document(_doc("doc_a", "License"))
    repo.upsert_document(_doc("doc_b", "Service"))
    svc.log_query(_log(query_id="q1", cited_doc_ids=["doc_a"], route="corpus_technical"))
    svc.log_query(
        _log(
            query_id="q2",
            question="Force majeure?",
            cited_doc_ids=["doc_b"],
            route="general",
            config_hash="def456",
        )
    )
    svc.rate("q1", -1, tags=["wrong_answer"])
    svc.rate("q1", -1, tags=["hallucination"])
    svc.rate("q2", 1, tags=["good"])

    by_route = {row["key"]: row for row in ratings_by_route(repo)}
    assert by_route["corpus_technical"]["n"] == 2
    assert by_route["corpus_technical"]["mean_rating"] == -1
    assert by_route["general"]["up"] == 1

    by_type = {row["key"]: row for row in ratings_by_agreement_type(repo)}
    assert by_type["License"]["down"] == 2
    assert by_type["Service"]["up"] == 1

    by_hash = {row["key"]: row for row in ratings_by_config_hash(repo)}
    assert set(by_hash) == {"abc123", "def456"}

    worst = worst_queries(repo, n=1)
    assert worst[0]["query_id"] == "q1"

    csv_path = export_csv(repo, tmp_path / "feedback.csv")
    text = csv_path.read_text(encoding="utf-8")
    assert "wrong_answer" in text
    assert "License" in text
    repo.close()
