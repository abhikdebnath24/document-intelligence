from __future__ import annotations

from collections.abc import Sequence

from docintel.core.interfaces import BaseReranker
from docintel.core.types import RetrievedChunk


class NoOpReranker(BaseReranker):
    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        _ = query
        out: list[RetrievedChunk] = []
        for rank, row in enumerate(list(chunks)[: max(0, top_n)], start=1):
            out.append(
                row.model_copy(
                    update={
                        "rank": rank,
                        "source": "reranked",
                        "provenance": [*row.provenance, "none"],
                    }
                )
            )
        return out
