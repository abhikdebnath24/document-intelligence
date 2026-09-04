from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from docintel.core.device import resolve_device
from docintel.core.interfaces import BaseReranker
from docintel.core.types import RetrievedChunk
from docintel.settings import hf_token


class NoOpReranker(BaseReranker):
    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        _ = query
        return _cut(chunks, top_n, "none")


class CrossEncoderReranker(BaseReranker):
    """sentence-transformers CrossEncoder. Default model is bge-reranker-base."""

    def __init__(
        self,
        model_id: str = "BAAI/bge-reranker-base",
        top_n: int = 10,
        device: str = "auto",
        batch_size: int = 16,
        **_: object,
    ) -> None:
        self.model_id = model_id
        self.top_n = max(1, int(top_n))
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self._model: Any | None = None

    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        keep = min(self.top_n, max(0, top_n), len(chunks))
        if keep == 0:
            return []
        model = self._load()
        pairs = [[query, row.chunk.text] for row in chunks]
        scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        ranked = sorted(
            zip(chunks, scores, strict=True),
            key=lambda item: (-float(item[1]), item[0].chunk.chunk_id),
        )
        out: list[RetrievedChunk] = []
        for rank, (row, score) in enumerate(ranked[:keep], start=1):
            out.append(
                row.model_copy(
                    update={
                        "score": float(score),
                        "rank": rank,
                        "source": "reranked",
                        "provenance": [*row.provenance, "cross_encoder"],
                    }
                )
            )
        return out

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            token = hf_token()
            kwargs: dict[str, object] = {"device": resolve_device(self.device)}
            if token:
                kwargs["token"] = token
            self._model = CrossEncoder(self.model_id, **kwargs)
        return self._model


def _cut(chunks: Sequence[RetrievedChunk], top_n: int, tag: str) -> list[RetrievedChunk]:
    out: list[RetrievedChunk] = []
    for rank, row in enumerate(list(chunks)[: max(0, top_n)], start=1):
        out.append(
            row.model_copy(
                update={
                    "rank": rank,
                    "source": "reranked",
                    "provenance": [*row.provenance, tag],
                }
            )
        )
    return out
