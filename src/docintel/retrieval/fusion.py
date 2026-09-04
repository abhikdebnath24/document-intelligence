from __future__ import annotations

import math
from collections.abc import Sequence

from docintel.core.interfaces import BaseFusion
from docintel.core.types import RetrievedChunk


def _key(chunk: RetrievedChunk) -> str:
    return chunk.chunk.chunk_id or f"{chunk.chunk.doc_id}:{chunk.chunk.chunk_idx}"


class _Accumulator:
    """Sum per-chunk contributions; keep one payload row and the union of provenance."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.scores: dict[str, float] = {}
        self.rows: dict[str, RetrievedChunk] = {}
        self.prov: dict[str, list[str]] = {}

    def add(self, row: RetrievedChunk, contribution: float) -> None:
        kid = _key(row)
        self.scores[kid] = self.scores.get(kid, 0.0) + contribution
        self.rows.setdefault(kid, row)
        seen = self.prov.setdefault(kid, [])
        for stage in row.provenance:
            if stage not in seen:
                seen.append(stage)

    def result(self) -> list[RetrievedChunk]:
        merged = {
            kid: row.model_copy(
                update={"score": self.scores[kid], "provenance": [*self.prov[kid], self.tag]}
            )
            for kid, row in self.rows.items()
        }
        ordered = sorted(merged.values(), key=lambda r: (-r.score, _key(r)))
        return [
            row.model_copy(update={"rank": rank, "source": "fused"})
            for rank, row in enumerate(ordered, start=1)
        ]


class RRFFusion(BaseFusion):
    """Cormack RRF. Rank is 1-based (RetrievedChunk.rank). Qdrant native uses 0-based."""

    def __init__(self, k: int = 60, **_: object) -> None:
        self.k = max(1, int(k))

    def fuse(self, ranked_lists: Sequence[Sequence[RetrievedChunk]]) -> list[RetrievedChunk]:
        acc = _Accumulator("rrf")
        for rows in ranked_lists:
            for row in rows:
                acc.add(row, 1.0 / (self.k + row.rank))
        return acc.result()


class DBSFFusion(BaseFusion):
    """Qdrant DBSF: hat{s} = (s - (mu - 3 sig)) / (6 sig); identical scores -> 0.5."""

    def fuse(self, ranked_lists: Sequence[Sequence[RetrievedChunk]]) -> list[RetrievedChunk]:
        acc = _Accumulator("dbsf")
        for rows in ranked_lists:
            mapped = _dbsf_normalize([row.score for row in rows])
            for row, hat in zip(rows, mapped, strict=True):
                acc.add(row, hat)
        return acc.result()


class WeightedFusion(BaseFusion):
    def __init__(self, alpha: float = 0.5, **_: object) -> None:
        self.alpha = float(alpha)

    def fuse(self, ranked_lists: Sequence[Sequence[RetrievedChunk]]) -> list[RetrievedChunk]:
        if len(ranked_lists) != 2:
            raise ValueError("weighted fusion expects exactly two ranked lists")
        dense, sparse = ranked_lists
        acc = _Accumulator("weighted")
        for row, val in zip(dense, _minmax([r.score for r in dense]), strict=True):
            acc.add(row, self.alpha * val)
        for row, val in zip(sparse, _minmax([r.score for r in sparse]), strict=True):
            acc.add(row, (1.0 - self.alpha) * val)
        return acc.result()


def _dbsf_normalize(values: list[float]) -> list[float]:
    n = len(values)
    if n == 0:
        return []
    if n == 1 or max(values) == min(values):
        return [0.5] * n
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    sigma = math.sqrt(var)
    if sigma == 0:
        return [0.5] * n
    lo = mean - 3 * sigma
    return [(v - lo) / (6 * sigma) for v in values]


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]
