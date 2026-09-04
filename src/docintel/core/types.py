from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Route = Literal["general", "corpus_technical", "ambiguous", "out_of_scope"]
RetrievalSource = Literal["dense", "sparse", "fused", "reranked"]


class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class TextBlock(BaseModel):
    text: str
    bbox: BBox


class Page(BaseModel):
    page_no: int
    text: str
    blocks: list[TextBlock] = Field(default_factory=list)


class Document(BaseModel):
    doc_id: str
    source_path: str
    agreement_type: str
    sha256: str
    pages: list[Page] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    page_start: int
    page_end: int
    bboxes: list[BBox] = Field(default_factory=list)
    section_header: str | None = None
    parent_id: str | None = None
    chunk_idx: int
    char_span: tuple[int, int] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SparseVector(BaseModel):
    indices: list[int]
    values: list[float]


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    source: RetrievalSource
    rank: int
    provenance: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    page_no: int
    bboxes: list[BBox] = Field(default_factory=list)
    quote: str


class Timing(BaseModel):
    node: str
    t_ms: float


class Answer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    route: Route
    abstained: bool = False
    groundedness: float | None = None
    trace_id: str | None = None
    timings: list[Timing] = Field(default_factory=list)


class RetrievalQuery(BaseModel):
    text: str
    k: int = 20
    filters: dict[str, Any] = Field(default_factory=dict)
    doc_id: str | None = None


class GradeResult(BaseModel):
    chunk_id: str
    relevant: bool
    reason: str = ""


class GroundednessResult(BaseModel):
    grounded: bool
    score: float
    unsupported_claims: list[str] = Field(default_factory=list)


class DocumentRecord(BaseModel):
    doc_id: str
    source_path: str
    agreement_type: str
    sha256: str
    pipeline_version: str
    index_sig: str
    n_chunks: int
    collection: str
    status: Literal["pending", "indexed", "failed"]
    split: str | None = None


class QueryLog(BaseModel):
    query_id: str
    session_id: str | None = None
    question: str
    route: Route | None = None
    config_hash: str
    profile: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    answer: str
    abstained: bool = False
    groundedness: float | None = None
    rewrites: int = 0
    latency_ms: int = 0
    token_usage: dict[str, int] = Field(default_factory=dict)
    trace_path: str | None = None


class Feedback(BaseModel):
    feedback_id: str
    query_id: str
    rating: int
    tags: list[str] = Field(default_factory=list)
    comment: str | None = None
