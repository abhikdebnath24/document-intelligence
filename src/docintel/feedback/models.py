from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Ingest DocumentRegistry already owns this table in the same SQLite file.
# Column set must stay identical so create_all is a no-op on an ingested DB.
FEEDBACK_TAGS = frozenset(
    {
        "wrong_answer",
        "hallucination",
        "wrong_citation",
        "missing_citation",
        "incomplete",
        "should_have_abstained",
        "should_not_have_abstained",
        "good",
    }
)
ALLOWED_RATINGS = frozenset({-1, 1, 2, 3, 4, 5})


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String, primary_key=True)
    source_path: Mapped[str] = mapped_column(String, nullable=False)
    agreement_type: Mapped[str] = mapped_column(String, nullable=False)
    sha256: Mapped[str] = mapped_column(String, nullable=False)
    pipeline_version: Mapped[str] = mapped_column(String, nullable=False)
    index_sig: Mapped[str] = mapped_column(String, nullable=False)
    n_chunks: Mapped[int] = mapped_column(Integer, nullable=False)
    collection: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    split: Mapped[str | None] = mapped_column(String, nullable=True)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


class QueryLogRow(Base):
    __tablename__ = "query_logs"

    query_id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False, default=utc_now)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    route: Mapped[str | None] = mapped_column(String, nullable=True)
    config_hash: Mapped[str] = mapped_column(String, nullable=False)
    profile: Mapped[str] = mapped_column(String, nullable=False)
    retrieved_chunk_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    cited_chunk_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    cited_doc_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    abstained: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    groundedness: Mapped[float | None] = mapped_column(Float, nullable=True)
    rewrites: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    token_usage: Mapped[dict[str, int]] = mapped_column(JSON, nullable=False, default=dict)
    trace_path: Mapped[str | None] = mapped_column(String, nullable=True)


class FeedbackRow(Base):
    __tablename__ = "feedback"

    feedback_id: Mapped[str] = mapped_column(String, primary_key=True)
    query_id: Mapped[str] = mapped_column(ForeignKey("query_logs.query_id"), nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False, default=utc_now)
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    tags: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    corrected_citation: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
