from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from docintel.core.interfaces import BaseFeedbackRepository
from docintel.core.types import DocumentRecord, Feedback, QueryLog
from docintel.feedback.models import Base, DocumentRow, FeedbackRow, QueryLogRow, utc_now


def resolve_db_url(url: str) -> str:
    """Same file the ingest `DocumentRegistry` opens: relative sqlite paths resolve against
    the CWD, like `.qdrant`. Both sides must agree or `documents` joins come back empty."""
    if ":memory:" in url:
        return "sqlite:///:memory:"
    if not url.startswith("sqlite:///"):
        return url
    return "sqlite:///" + str(Path(url.removeprefix("sqlite:///")).resolve())


def make_engine(url: str) -> Engine:
    resolved = resolve_db_url(url)
    if resolved == "sqlite:///:memory:":
        engine = create_engine(
            resolved,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    else:
        if resolved.startswith("sqlite:///"):
            Path(resolved.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(resolved)
    Base.metadata.create_all(engine)
    return engine


class SqlAlchemyFeedbackRepository(BaseFeedbackRepository):
    def __init__(self, db_url: str) -> None:
        self.db_url = resolve_db_url(db_url)
        self.engine = make_engine(db_url)
        self._session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def session(self) -> Session:
        return self._session_factory()

    def close(self) -> None:
        self.engine.dispose()

    def log_query(self, log: QueryLog) -> str:
        row = QueryLogRow(
            query_id=log.query_id,
            session_id=log.session_id,
            created_at=utc_now(),
            question=log.question,
            route=log.route,
            config_hash=log.config_hash,
            profile=log.profile,
            retrieved_chunk_ids=list(log.retrieved_chunk_ids),
            cited_chunk_ids=list(log.cited_chunk_ids),
            cited_doc_ids=list(log.cited_doc_ids),
            answer=log.answer,
            abstained=log.abstained,
            groundedness=log.groundedness,
            rewrites=log.rewrites,
            latency_ms=log.latency_ms,
            token_usage=dict(log.token_usage),
            trace_path=log.trace_path,
        )
        with self.session() as session:
            session.merge(row)
            session.commit()
        return log.query_id

    def add_feedback(self, feedback: Feedback) -> None:
        with self.session() as session:
            if session.get(QueryLogRow, feedback.query_id) is None:
                raise ValueError(f"unknown query_id {feedback.query_id!r}")
            session.merge(
                FeedbackRow(
                    feedback_id=feedback.feedback_id,
                    query_id=feedback.query_id,
                    created_at=utc_now(),
                    rating=feedback.rating,
                    tags=list(feedback.tags),
                    comment=feedback.comment,
                    corrected_citation=feedback.corrected_citation,
                )
            )
            session.commit()

    def upsert_document(self, record: DocumentRecord) -> None:
        with self.session() as session:
            session.merge(
                DocumentRow(
                    doc_id=record.doc_id,
                    source_path=record.source_path,
                    agreement_type=record.agreement_type,
                    sha256=record.sha256,
                    pipeline_version=record.pipeline_version,
                    index_sig=record.index_sig,
                    n_chunks=record.n_chunks,
                    collection=record.collection,
                    status=record.status,
                    split=record.split,
                    updated_at=utc_now(),
                )
            )
            session.commit()

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        with self.session() as session:
            row = session.get(DocumentRow, doc_id)
            if row is None:
                return None
            return DocumentRecord(
                doc_id=row.doc_id,
                source_path=row.source_path,
                agreement_type=row.agreement_type,
                sha256=row.sha256,
                pipeline_version=row.pipeline_version,
                index_sig=row.index_sig,
                n_chunks=row.n_chunks,
                collection=row.collection,
                status=row.status,  # type: ignore[arg-type]
                split=row.split,
            )

    def get_query(self, query_id: str) -> QueryLog | None:
        with self.session() as session:
            row = session.get(QueryLogRow, query_id)
            return _to_query_log(row) if row is not None else None

    def list_feedback(self, query_id: str | None = None) -> list[Feedback]:
        with self.session() as session:
            stmt = select(FeedbackRow)
            if query_id is not None:
                stmt = stmt.where(FeedbackRow.query_id == query_id)
            return [_to_feedback(row) for row in session.scalars(stmt)]


def _to_query_log(row: QueryLogRow) -> QueryLog:
    return QueryLog(
        query_id=row.query_id,
        session_id=row.session_id,
        question=row.question,
        route=row.route,  # type: ignore[arg-type]
        config_hash=row.config_hash,
        profile=row.profile,
        retrieved_chunk_ids=list(row.retrieved_chunk_ids or []),
        cited_chunk_ids=list(row.cited_chunk_ids or []),
        cited_doc_ids=list(row.cited_doc_ids or []),
        answer=row.answer,
        abstained=row.abstained,
        groundedness=row.groundedness,
        rewrites=row.rewrites,
        latency_ms=row.latency_ms,
        token_usage=dict(row.token_usage or {}),
        trace_path=row.trace_path,
    )


def _to_feedback(row: FeedbackRow) -> Feedback:
    return Feedback(
        feedback_id=row.feedback_id,
        query_id=row.query_id,
        rating=row.rating,
        tags=list(row.tags or []),
        comment=row.comment,
        corrected_citation=row.corrected_citation,
    )
