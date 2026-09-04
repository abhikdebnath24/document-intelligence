from __future__ import annotations

from docintel.core.types import Chunk, RetrievedChunk
from docintel.retrieval.fusion import DBSFFusion, RRFFusion, WeightedFusion


def _row(cid: str, score: float, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=cid, doc_id="d", text=cid, page_start=1, page_end=1, chunk_idx=0),
        score=score,
        source="dense",
        rank=rank,
    )


def test_rrf_prefers_overlap_and_tie_breaks_on_chunk_id() -> None:
    a = [_row("x", 0.9, 1), _row("y", 0.8, 2)]
    b = [_row("y", 0.5, 1), _row("x", 0.4, 2)]
    tie = RRFFusion(k=60).fuse([a, b])
    # 1/61 + 1/62 for both -> chunk_id order
    assert [r.chunk.chunk_id for r in tie] == ["x", "y"]
    assert tie[0].score == tie[1].score
    assert tie[0].source == "fused"
    assert tie[0].provenance == ["rrf"]

    overlap = RRFFusion(k=60).fuse([[_row("x", 0.9, 1), _row("y", 0.8, 2)], [_row("y", 0.5, 1)]])
    assert [r.chunk.chunk_id for r in overlap] == ["y", "x"]
    assert [r.rank for r in overlap] == [1, 2]


def test_provenance_is_union_of_lists() -> None:
    dense = _row("x", 0.9, 1).model_copy(update={"provenance": ["dense"]})
    sparse = _row("x", 3.0, 1).model_copy(update={"provenance": ["sparse"]})
    fused = RRFFusion().fuse([[dense], [sparse]])
    assert fused[0].provenance == ["dense", "sparse", "rrf"]


def test_dbsf_identical_scores_are_half() -> None:
    rows = [_row("a", 2.0, 1), _row("b", 2.0, 2)]
    fused = DBSFFusion().fuse([rows])
    assert all(abs(r.score - 0.5) < 1e-9 for r in fused)


def test_weighted_requires_two_lists() -> None:
    try:
        WeightedFusion().fuse([[_row("a", 1.0, 1)]])
    except ValueError as exc:
        assert "two" in str(exc)
    else:
        raise AssertionError("expected ValueError")
    fused = WeightedFusion(alpha=1.0).fuse([[_row("a", 1.0, 1)], [_row("b", 9.0, 1)]])
    assert fused[0].chunk.chunk_id == "a"
