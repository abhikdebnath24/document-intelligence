from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from docintel.core.types import (
    Chunk,
    Document,
    DocumentRecord,
    Feedback,
    GradeResult,
    GroundednessResult,
    QueryLog,
    RetrievalQuery,
    RetrievedChunk,
    SparseVector,
)


class BaseLoader(ABC):
    @abstractmethod
    def load(self, path: Path) -> Document: ...

    @abstractmethod
    def supports(self, path: Path) -> bool: ...

    def alias(self) -> str:
        return self.__class__.__name__


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]: ...

    @abstractmethod
    def alias(self) -> str: ...


class BaseDenseEmbedder(ABC):
    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...

    @abstractmethod
    def alias(self) -> str: ...


class BaseSparseEncoder(ABC):
    @abstractmethod
    def encode_documents(self, texts: Sequence[str]) -> list[SparseVector]: ...

    @abstractmethod
    def encode_query(self, text: str) -> SparseVector: ...

    def alias(self) -> str:
        return self.__class__.__name__


class BaseVectorStore(ABC):
    @abstractmethod
    def ensure_collection(
        self, name: str, dense_dim: int, has_sparse: bool, index_sig: str
    ) -> None: ...

    @abstractmethod
    def upsert(self, points: Sequence[dict[str, Any]]) -> None: ...

    @abstractmethod
    def delete_by_doc(self, doc_id: str, *, except_hash: str | None = None) -> int: ...

    @abstractmethod
    def count_by_doc_hash(self, doc_id: str, content_hash: str) -> int: ...

    @abstractmethod
    def search_dense(
        self, q: Sequence[float], k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]: ...

    @abstractmethod
    def search_sparse(
        self, q: SparseVector, k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]: ...

    @abstractmethod
    def search_hybrid(
        self,
        qd: Sequence[float],
        qs: SparseVector,
        k: int,
        fusion: str,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]: ...


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]: ...


class BaseFusion(ABC):
    @abstractmethod
    def fuse(self, ranked_lists: Sequence[Sequence[RetrievedChunk]]) -> list[RetrievedChunk]: ...


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]: ...


class BaseQueryTransform(ABC):
    @abstractmethod
    def transform(self, query: str) -> list[str]: ...


class BaseGrader(ABC):
    @abstractmethod
    def grade(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[GradeResult]: ...


class BaseGroundednessVerifier(ABC):
    @abstractmethod
    def verify(self, answer: str, chunks: Sequence[RetrievedChunk]) -> GroundednessResult: ...


class BaseFeedbackRepository(ABC):
    @abstractmethod
    def log_query(self, log: QueryLog) -> str: ...

    @abstractmethod
    def add_feedback(self, feedback: Feedback) -> None: ...

    @abstractmethod
    def upsert_document(self, record: DocumentRecord) -> None: ...

    @abstractmethod
    def get_document(self, doc_id: str) -> DocumentRecord | None: ...
