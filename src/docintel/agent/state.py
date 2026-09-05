from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict

from docintel.core.types import Route


class RAGState(TypedDict, total=False):
    question: str
    history: list[dict[str, str]]
    route: Route
    route_reason: str
    doc_hint: str
    filters: dict[str, Any]
    search_queries: list[str]
    candidate_ids: list[str]
    ranked_ids: list[str]
    grades: list[dict[str, Any]]
    relevant_ids: list[str]
    rewrites: int
    regen_count: int
    draft_answer: dict[str, Any]
    grounded: bool
    groundedness_score: float
    unsupported_claims: list[str]
    answer: dict[str, Any]
    timings: Annotated[list[dict[str, Any]], add]
    trace: Annotated[list[dict[str, Any]], add]
