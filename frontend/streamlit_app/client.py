from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from docintel.agent.graph import GraphTick
from docintel.config import AppConfig
from docintel.core.logging import get_logger
from docintel.core.types import DocumentRecord, Feedback, QueryLog
from docintel.ingestion.pipeline import IngestReport
from docintel.ingestion.registry_store import DocumentRegistry
from docintel.service.container import Container
from docintel.service.feedback_service import FeedbackService
from docintel.service.ingest_service import IngestService
from docintel.service.query_service import QueryService

log_ = get_logger(__name__)


class RagClient(Protocol):
    config: AppConfig
    repo_root: Path

    def ask_iter(
        self,
        question: str,
        *,
        session_id: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[GraphTick]: ...

    def rate(
        self,
        query_id: str,
        rating: int,
        *,
        tags: list[str] | None = None,
        comment: str | None = None,
    ) -> Feedback: ...

    def list_documents(self) -> list[DocumentRecord]: ...

    def ingest_paths(self, paths: list[Path]) -> IngestReport: ...

    def find_source(self, doc_id: str) -> Path | None: ...

    def close(self) -> None: ...


class InProcessClient:
    def __init__(self, config: AppConfig, feedback: FeedbackService, repo_root: Path) -> None:
        self.config = config
        self.repo_root = repo_root
        self.feedback = feedback
        self.ingest = IngestService(config, repo_root=repo_root)
        self._query: QueryService | None = None

    @property
    def query(self) -> QueryService:
        if self._query is None:
            self._query = QueryService(Container(self.config, repo_root=self.repo_root))
        return self._query

    def ask_iter(
        self,
        question: str,
        *,
        session_id: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[GraphTick]:
        for tick in self.query.ask_iter(question, session_id=session_id, history=history):
            if tick.kind == "done" and tick.state is not None:
                log: QueryLog = tick.state["log"]
                try:
                    self.feedback.log_query(log)
                except Exception as exc:  # a locked/missing sqlite must not eat the answer
                    log_.warning("query_log_failed", query_id=log.query_id, error=str(exc))
            yield tick

    def rate(
        self,
        query_id: str,
        rating: int,
        *,
        tags: list[str] | None = None,
        comment: str | None = None,
    ) -> Feedback:
        return self.feedback.rate(
            query_id,
            rating,
            tags=tags,
            comment=comment,
            feedback_id=f"streamlit-{query_id}",
        )

    def list_documents(self) -> list[DocumentRecord]:
        registry = DocumentRegistry(self.config.feedback.db_url)
        try:
            return registry.list_all()
        finally:
            registry.close()

    def ingest_paths(self, paths: list[Path]) -> IngestReport:
        # One embedded Qdrant lock: drop the query store before ingest writes.
        if self._query is not None:
            self._query.container.pipeline.close()
        return self.ingest.ingest_paths(paths)

    def find_source(self, doc_id: str) -> Path | None:
        from docintel.service.pdf_render import resolve_pdf_path

        registry = DocumentRegistry(self.config.feedback.db_url)
        try:
            rec = registry.get(doc_id)
        finally:
            registry.close()
        if rec is None:
            return None
        return resolve_pdf_path(rec.source_path, self.repo_root)

    def close(self) -> None:
        if self._query is not None:
            self._query.close()
            self._query = None
        closer = getattr(self.feedback.repo, "close", None)
        if closer:
            closer()


class HttpClient:
    def __init__(self, api_url: str) -> None:
        raise NotImplementedError(
            f"frontend.backend=http ({api_url}) is WS8. Use frontend.backend=inprocess."
        )
