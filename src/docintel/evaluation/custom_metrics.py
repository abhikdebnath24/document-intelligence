from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from docintel.evaluation.frameworks.base import EvalResult, EvalSample


def custom_metrics(samples: Sequence[EvalSample]) -> dict[str, float]:
    n = len(samples)
    if not n:
        return {
            "n": 0.0,
            "route_accuracy": 0.0,
            "abstention_precision": 0.0,
            "abstention_recall": 0.0,
            "citation_validity": 0.0,
            "mean_groundedness": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "llm_calls_per_query": 0.0,
            "tokens_per_query": 0.0,
        }
    tp = sum(1 for s in samples if s.expected_abstain and s.abstained)
    fp = sum(1 for s in samples if s.abstained and not s.expected_abstain)
    fn = sum(1 for s in samples if s.expected_abstain and not s.abstained)
    cites = [s.citations_valid for s in samples if s.citations_valid is not None]
    grounds = [s.groundedness for s in samples if s.groundedness is not None]
    lat = sorted(s.latency_ms for s in samples)
    out = {
        "n": float(n),
        "route_accuracy": sum(1 for s in samples if s.predicted_route == s.expected_route) / n,
        "abstention_precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "abstention_recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "citation_validity": (sum(1 for ok in cites if ok) / len(cites)) if cites else 0.0,
        "mean_groundedness": (sum(grounds) / len(grounds)) if grounds else 0.0,
        "latency_p50_ms": _percentile(lat, 50),
        "latency_p95_ms": _percentile(lat, 95),
        "llm_calls_per_query": sum(s.llm_calls for s in samples) / n,
    }
    # token usage is only present when the caller records it; a fake 0 would mislead
    tokens = [sum(s.token_usage.values()) for s in samples if s.token_usage]
    if tokens:
        out["tokens_per_query"] = sum(tokens) / n
    return out


def framework_agreement(
    ragas: EvalResult, deepeval: EvalResult, *, delta: float = 0.3
) -> dict[str, Any]:
    left = {
        str(row["id"]): float(row["faithfulness"])
        for row in ragas.per_sample
        if row.get("faithfulness") is not None
    }
    right = {
        str(row["id"]): float(row["faithfulness"])
        for row in deepeval.per_sample
        if row.get("faithfulness") is not None
    }
    ids = sorted(set(left) & set(right))
    pairs = [(left[i], right[i]) for i in ids]
    disagrees = [
        {"id": i, "ragas": left[i], "deepeval": right[i]}
        for i in ids
        if abs(left[i] - right[i]) > delta
    ]
    return {
        "n": len(pairs),
        "spearman": _spearman([a for a, _ in pairs], [b for _, b in pairs]),
        "disagree_gt_0.3": disagrees,
        "headline": "ragas",
    }


def agreement_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Framework agreement",
        "",
        "RAGAS faithfulness is the headline. DeepEval is a cross-check.",
        "",
        f"- n={report['n']}",
        f"- spearman={report['spearman']:.3f}",
        f"- disagreements>|0.3|={len(report['disagree_gt_0.3'])}",
        "",
    ]
    if report["disagree_gt_0.3"]:
        lines += ["| id | ragas | deepeval |", "|---|---:|---:|"]
        for row in report["disagree_gt_0.3"]:
            lines.append(f"| {row['id']} | {row['ragas']:.3f} | {row['deepeval']:.3f} |")
        lines.append("")
    return "\n".join(lines)


def tag_miss(rec: dict[str, Any]) -> str:
    gold = rec.get("doc_id") or rec.get("doc_stem")
    docs = rec.get("doc_ids") or []
    if gold and gold not in docs:
        return "exact_term_miss"
    if float(rec.get("hit@10") or 0) == 0:
        return "chunk_boundary"
    if float(rec.get("hit@5") or 0) == 0 and float(rec.get("hit@10") or 0) > 0:
        return "reranker_demoted_gold"
    return "header_noise"


def top_misses(rows: Sequence[dict[str, Any]], *, n: int = 20) -> list[dict[str, Any]]:
    scored = [r for r in rows if float(r.get("hit@10") or 0) == 0 or float(r.get("r@10") or 0) < 1]
    scored = sorted(scored, key=lambda r: (float(r.get("r@10") or 0), float(r.get("mrr") or 0)))
    out: list[dict[str, Any]] = []
    for rec in scored[:n]:
        out.append(
            {
                "id": rec.get("id"),
                "bucket": rec.get("bucket"),
                "doc_id": rec.get("doc_id"),
                "r@10": rec.get("r@10"),
                "hit@10": rec.get("hit@10"),
                "tag": tag_miss(rec),
            }
        )
    return out


def _percentile(sorted_vals: Sequence[int | float], p: int) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    idx = min(len(sorted_vals) - 1, max(0, round((p / 100) * (len(sorted_vals) - 1))))
    return float(sorted_vals[idx])


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    rx, ry = _ranks(xs), _ranks(ys)
    return _pearson(rx, ry)


def _ranks(vals: Sequence[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return float(num / (dx * dy))
