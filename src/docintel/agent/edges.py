from __future__ import annotations

from typing import Literal

from docintel.agent.state import RAGState

AfterClassify = Literal["answer_general", "clarify", "refuse", "plan_retrieval"]
AfterGrade = Literal["generate", "rewrite_query", "abstain"]
AfterVerify = Literal["finalize", "regenerate_strict", "abstain"]


def after_classify(state: RAGState) -> AfterClassify:
    route = state.get("route") or "corpus_technical"
    if route == "general":
        return "answer_general"
    if route == "ambiguous":
        return "clarify"
    if route == "out_of_scope":
        return "refuse"
    return "plan_retrieval"


def after_grade(state: RAGState, *, min_relevant: int, max_rewrites: int) -> AfterGrade:
    """min_relevant triggers a rewrite. Abstain only with zero evidence after the cap;
    one relevant chunk after exhausted rewrites still answers."""
    n = len(state.get("relevant_ids") or [])
    if n >= min_relevant:
        return "generate"
    if int(state.get("rewrites") or 0) < max_rewrites:
        return "rewrite_query"
    return "generate" if n > 0 else "abstain"


def after_verify(state: RAGState) -> AfterVerify:
    if state.get("grounded"):
        return "finalize"
    if int(state.get("regen_count") or 0) < 1:
        return "regenerate_strict"
    return "abstain"
