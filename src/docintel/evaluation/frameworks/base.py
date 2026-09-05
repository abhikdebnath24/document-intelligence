from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from docintel.core.types import Answer, QueryLog
from docintel.evaluation.gold import QAItem, SpanMatcher


class EvalSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    retrieved_contexts: list[str] = Field(default_factory=list)
    answer: str
    reference: str | None = None
    expected_abstain: bool = False
    expected_route: str = "corpus_technical"
    predicted_route: str = "corpus_technical"
    abstained: bool = False
    citations_valid: bool | None = None
    latency_ms: int = 0
    llm_calls: int = 0
    token_usage: dict[str, int] = Field(default_factory=dict)
    groundedness: float | None = None
    bucket: str = "slot"


class EvalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    framework: str
    overall: dict[str, float] = Field(default_factory=dict)
    per_sample: list[dict[str, Any]] = Field(default_factory=list)
    skipped_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class BaseGenerationEvaluator(ABC):
    @abstractmethod
    def evaluate(self, samples: list[EvalSample]) -> EvalResult: ...


def samples_from_records(
    items: list[QAItem],
    records: list[dict[str, Any]],
    *,
    match_threshold: int = 90,
) -> list[EvalSample]:
    by_id = {item.id: item for item in items}
    matcher = SpanMatcher(match_threshold)
    out: list[EvalSample] = []
    for rec in records:
        item = by_id.get(str(rec.get("id") or ""))
        if item is None:
            continue
        contexts = [str(c) for c in (rec.get("retrieved_contexts") or []) if c]
        cites = rec.get("citations") or []
        out.append(
            EvalSample(
                id=item.id,
                question=item.question,
                retrieved_contexts=contexts,
                answer=str(rec.get("answer") or ""),
                reference=_reference(item),
                expected_abstain=item.expected_abstain,
                expected_route=item.expected_route,
                predicted_route=str(rec.get("route") or "corpus_technical"),
                abstained=bool(rec.get("abstained")),
                citations_valid=_cites_ok(cites, contexts, matcher),
                latency_ms=int(rec.get("latency_ms") or 0),
                llm_calls=int(rec.get("llm_calls") or 0),
                token_usage={k: int(v) for k, v in (rec.get("token_usage") or {}).items()},
                groundedness=rec.get("groundedness"),
                bucket=item.bucket,
            )
        )
    return out


def record_from_ask(item: QAItem, answer: Answer, log: QueryLog) -> dict[str, Any]:
    return {
        "id": item.id,
        "bucket": item.bucket,
        "question": item.question,
        "answer": answer.text,
        "route": answer.route,
        "abstained": answer.abstained,
        "expected_abstain": item.expected_abstain,
        "expected_route": item.expected_route,
        "retrieved_contexts": list(log.retrieved_contexts),
        "citations": [c.model_dump() for c in answer.citations],
        "latency_ms": log.latency_ms,
        "llm_calls": log.llm_calls,
        "token_usage": dict(log.token_usage),
        "groundedness": answer.groundedness,
        "rewrites": log.rewrites,
        "query_id": log.query_id,
        "trace_path": log.trace_path,
    }


def _reference(item: QAItem) -> str | None:
    if item.gold_answer:
        return item.gold_answer
    if item.gold_spans:
        return " ".join(item.gold_spans)
    return None


def _cites_ok(cites: list[Any], contexts: list[str], matcher: SpanMatcher) -> bool | None:
    if not cites:
        return None
    hay = "\n".join(contexts)
    for cite in cites:
        quote = cite.get("quote") if isinstance(cite, dict) else getattr(cite, "quote", "")
        if not quote or not matcher.fragment_in_text(str(quote), hay):
            return False
    return True
