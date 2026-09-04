from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from docintel.core.device import resolve_device
from docintel.core.interfaces import BaseReranker
from docintel.core.types import RetrievedChunk


class NoOpReranker(BaseReranker):
    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        _ = query
        return _cut(chunks, top_n, "none")


class CrossEncoderReranker(BaseReranker):
    """sentence-transformers CrossEncoder. Default model is mxbai-rerank-xsmall-v1."""

    def __init__(
        self,
        model_id: str = "mixedbread-ai/mxbai-rerank-xsmall-v1",
        top_n: int = 10,
        device: str = "auto",
        batch_size: int = 16,
        max_passage_tokens: int = 0,
        **_: object,
    ) -> None:
        self.model_id = model_id
        self.top_n = max(1, int(top_n))
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.max_passage_tokens = max(0, int(max_passage_tokens))
        self._model: Any | None = None

    def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        keep = min(self.top_n, max(0, top_n), len(chunks))
        if keep == 0:
            return []
        model = self._load()
        # One CE pair per window; chunk score = max over its windows so a gold span
        # near the chunk tail is still visible to a 512-token cross-encoder.
        pairs: list[list[str]] = []
        owner: list[int] = []
        for i, row in enumerate(chunks):
            for window in self._windows(model, row.chunk.text):
                pairs.append([query, window])
                owner.append(i)
        raw = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        scores = [float("-inf")] * len(chunks)
        for i, s in zip(owner, raw, strict=True):
            scores[i] = max(scores[i], float(s))
        ranked = sorted(
            zip(chunks, scores, strict=True),
            key=lambda item: (-item[1], item[0].chunk.chunk_id),
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

            self._model = CrossEncoder(self.model_id, device=resolve_device(self.device))
        return self._model

    def _windows(self, model: Any, text: str) -> list[str]:
        """Overlapping windows of `max_passage_tokens` (stride n/2). 0 = whole chunk."""
        n = self.max_passage_tokens
        if n == 0:
            return [text]
        tok = getattr(model, "tokenizer", None)
        if tok is not None:
            units: list[Any] = tok.encode(text, add_special_tokens=False, verbose=False)
        else:
            # ponytail: whitespace units when the CrossEncoder tokenizer is absent (unit tests)
            units = text.split()
        step = max(1, n // 2)
        out: list[str] = []
        start = 0
        while True:
            piece = units[start : start + n]
            if tok is not None:
                out.append(str(tok.decode(piece, skip_special_tokens=True)))
            else:
                out.append(" ".join(piece))
            if start + n >= len(units):
                return out
            start += step


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
