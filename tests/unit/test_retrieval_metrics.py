from __future__ import annotations

from docintel.core.types import Chunk, RetrievedChunk
from docintel.evaluation.gold import QAItem, SpanMatcher
from docintel.evaluation.retrieval_metrics import include_item, ndcg, question_metrics


def _chunk(text: str, doc_id: str = "doc-a", cid: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=cid, doc_id=doc_id, text=text, page_start=1, page_end=1, chunk_idx=0),
        score=1.0,
        source="dense",
        rank=1,
    )


def _item(**kwargs: object) -> QAItem:
    base: dict[str, object] = {
        "id": "q",
        "doc_stem": "doc-a",
        "agreement_type": "Service",
        "category": "Governing Law",
        "bucket": "slot",
        "question": "which law?",
        "gold_spans": ["laws of Delaware"],
        "gold_answer": "Delaware",
        "expected_route": "corpus_technical",
        "expected_abstain": False,
        "split": "dev",
    }
    base.update(kwargs)
    return QAItem.model_validate(base)


def test_metrics_hit_and_wrong_doc_is_miss() -> None:
    matcher = SpanMatcher(90)
    item = _item()
    hit = question_metrics(item, [_chunk("This shall be the laws of Delaware.")], matcher, [1, 5])
    assert hit["hit@1"] == 1.0
    assert hit["r@1"] == 1.0
    miss = question_metrics(
        item, [_chunk("This shall be the laws of Delaware.", doc_id="other")], matcher, [1]
    )
    assert miss["hit@1"] == 0.0


def test_no_gold_and_duplicates() -> None:
    assert include_item(_item(gold_spans=[], expected_abstain=True, bucket="no_answer")) is False
    assert include_item(_item(bucket="general", gold_spans=[])) is False
    assert ndcg([]) == 0.0
    assert ndcg([False, False]) == 0.0
    assert ndcg([True, False]) == 1.0


def test_multi_span_any_all() -> None:
    item = _item(gold_spans=["alpha clause", "beta clause"])
    matcher = SpanMatcher(90)
    rows = [
        _chunk("alpha clause is here", cid="1"),
        _chunk("unrelated", cid="2"),
    ]
    metrics = question_metrics(item, rows, matcher, [2])
    assert metrics["r@2_any"] == 1.0
    assert metrics["r@2_all"] == 0.0
    assert metrics["r@2"] == 0.5
    # a chunk holding one whole span is relevant; it must not need every span
    assert metrics["hit@2"] == 1.0
    assert metrics["p@2"] == 0.5
    assert metrics["mrr"] == 1.0

    split = [_chunk("alpha clause is here", cid="1"), _chunk("beta clause is here", cid="2")]
    both = question_metrics(item, split, matcher, [2])
    assert both["r@2_all"] == 1.0
    assert both["p@2"] == 1.0
