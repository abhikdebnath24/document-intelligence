from __future__ import annotations

from pydantic import BaseModel, Field

from docintel.core.types import Route


class RouteOut(BaseModel):
    route: Route
    reason: str = ""
    agreement_type: str | None = None
    doc_hint: str | None = None


class GradeItemOut(BaseModel):
    chunk_id: str
    relevant: bool
    reason: str = ""


class GradeBatchOut(BaseModel):
    grades: list[GradeItemOut] = Field(default_factory=list)


class RewriteOut(BaseModel):
    query: str
    reason: str = ""


class CiteDraft(BaseModel):
    chunk_id: str
    quote: str


class GenerateOut(BaseModel):
    answer: str
    citations: list[CiteDraft] = Field(default_factory=list)


class VerifyOut(BaseModel):
    grounded: bool
    score: float = 0.0
    unsupported_claims: list[str] = Field(default_factory=list)


class ClarifyOut(BaseModel):
    question: str
