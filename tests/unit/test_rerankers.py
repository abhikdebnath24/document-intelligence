from __future__ import annotations

from docintel.core.types import Chunk, RetrievedChunk
from docintel.retrieval.rerankers import CrossEncoderReranker, NoOpReranker


def _row(cid: str, text: str, score: float = 0.0, rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=cid, doc_id="d", text=text, page_start=1, page_end=1, chunk_idx=0),
        score=score,
        source="fused",
        rank=rank,
    )


def test_noop_keeps_order() -> None:
    rows = [_row("a", "first", 1.0, 1), _row("b", "second", 0.5, 2)]
    out = NoOpReranker().rerank("q", rows, 1)
    assert [r.chunk.chunk_id for r in out] == ["a"]
    assert out[0].provenance[-1] == "none"


def test_cross_encoder_reorders_by_score() -> None:
    rerank = CrossEncoderReranker(top_n=2)

    class _Fake:
        def predict(self, pairs: list[list[str]], **_: object) -> list[float]:
            return [0.1 if "wrong" in pair[1] else 0.9 for pair in pairs]

    rerank._model = _Fake()
    rows = [_row("a", "wrong clause", 1.0, 1), _row("b", "gold governing law", 0.2, 2)]
    out = rerank.rerank("governing law", rows, 10)
    assert [r.chunk.chunk_id for r in out] == ["b", "a"]
    assert out[0].source == "reranked"
    assert out[0].provenance[-1] == "cross_encoder"


def test_cross_encoder_windows_max_pool_and_keep_stored_text() -> None:
    rerank = CrossEncoderReranker(top_n=2, max_passage_tokens=3)
    seen: list[str] = []

    class _Fake:
        def predict(self, pairs: list[list[str]], **_: object) -> list[float]:
            seen.extend(pair[1] for pair in pairs)
            return [0.9 if "epsilon" in pair[1] else 0.1 for pair in pairs]

    rerank._model = _Fake()
    rows = [_row("head", "alpha beta gamma", 1.0, 1), _row("tail", "a b c d epsilon", 0.5, 2)]
    out = rerank.rerank("q", rows, 10)
    # tail chunk: windows of 3 with stride 1 cover the last token
    assert "c d epsilon" in seen
    assert [r.chunk.chunk_id for r in out] == ["tail", "head"]
    assert out[0].score == 0.9
    assert out[0].chunk.text == "a b c d epsilon"
