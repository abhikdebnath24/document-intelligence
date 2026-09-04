from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from docintel.core.types import RetrievedChunk
from docintel.evaluation.gold import QAItem, SpanMatcher


def chunk_relevant(item: QAItem, row: RetrievedChunk, matcher: SpanMatcher) -> bool:
    """Chunk is relevant when it holds at least one whole gold span from the gold doc.

    Multi-span items (31/52 in qa_dev+qa_test) rarely fit every span in one 512-token
    chunk; per-span coverage is reported separately as r@k / r@k_all.
    """
    if item.doc_stem and row.chunk.doc_id != item.doc_stem:
        return False
    return any(matcher.spans_in_text([span], row.chunk.text) for span in item.gold_spans)


def span_hits(item: QAItem, rows: Sequence[RetrievedChunk], matcher: SpanMatcher) -> list[bool]:
    hits: list[bool] = []
    for span in item.gold_spans:
        hits.append(
            any(
                row.chunk.doc_id == item.doc_stem and matcher.spans_in_text([span], row.chunk.text)
                for row in rows
            )
        )
    return hits


def question_metrics(
    item: QAItem,
    rows: Sequence[RetrievedChunk],
    matcher: SpanMatcher,
    ks: Sequence[int],
) -> dict[str, float]:
    labels = [chunk_relevant(item, row, matcher) for row in rows]
    out: dict[str, float] = {}
    first = next((i for i, ok in enumerate(labels, start=1) if ok), None)
    out["mrr"] = 0.0 if first is None else 1.0 / first
    for k in ks:
        top = labels[:k]
        rel = sum(1 for ok in top if ok)
        out[f"p@{k}"] = rel / k if k else 0.0
        out[f"hit@{k}"] = 1.0 if rel else 0.0
        out[f"ndcg@{k}"] = ndcg(top)
        found = span_hits(item, rows[:k], matcher)
        n_span = len(item.gold_spans)
        out[f"r@{k}"] = (sum(found) / n_span) if n_span else 0.0
        out[f"r@{k}_any"] = 1.0 if any(found) else 0.0
        out[f"r@{k}_all"] = 1.0 if found and all(found) else 0.0
    return out


def ndcg(labels: Sequence[bool]) -> float:
    if not labels:
        return 0.0
    dcg = sum((1.0 / math.log2(i + 1)) for i, ok in enumerate(labels, start=1) if ok)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, sum(1 for ok in labels if ok) + 1))
    if ideal == 0:
        return 0.0
    return dcg / ideal


def include_item(item: QAItem) -> bool:
    if item.bucket in {"general"} or item.expected_abstain:
        return False
    return bool(item.gold_spans)


def aggregate(
    rows: Sequence[dict[str, Any]],
    *,
    group_key: str | None = None,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(group_key) or "all") if group_key else "all"
        grouped[key].append(row)
    out: dict[str, Any] = {}
    for key, items in grouped.items():
        if not items:
            continue
        metric_keys = [k for k in items[0] if isinstance(items[0][k], float)]
        out[key] = {k: sum(item[k] for item in items) / len(items) for k in metric_keys}
        out[key]["n"] = float(len(items))
    return out
