from __future__ import annotations

import pytest

from docintel.evaluation.gold import SpanMatcher, gold_fragments, normalize_text


def test_normalize_joins_hyphenated_line_break() -> None:
    assert normalize_text("agree-\nment") == "agreement"
    assert normalize_text("agree- \n  ment") == "agreement"
    assert normalize_text("non-compete") == "non compete"
    assert normalize_text("Non-Compete") == normalize_text("non-compete")
    assert normalize_text("State  of   Nevada") == "state of nevada"


def test_omitted_splits_into_fragments() -> None:
    frags = gold_fragments("This Agreement <omitted> laws of the State of Nevada")
    assert frags == ["this agreement", "laws of the state of nevada"]


def test_matcher_whitespace_and_hyphenation() -> None:
    matcher = SpanMatcher(threshold=90)
    chunk = "This Agreement shall be governed by the laws of the State of Nevada."
    assert matcher.spans_in_text(["laws of the  State of Nevada"], chunk)
    assert matcher.spans_in_text(["gover-\nned by the laws"], chunk)


def test_matcher_requires_every_omitted_fragment() -> None:
    matcher = SpanMatcher(threshold=90)
    chunk = "This Agreement shall be governed by the laws of the State of Nevada."
    gold = ["This Agreement <omitted> laws of the State of Nevada"]
    assert matcher.spans_in_text(gold, chunk)
    missing_head = "The parties agree to arbitration under the laws of the State of Nevada."
    assert not matcher.spans_in_text(gold, missing_head)


def test_doc_id_mismatch_is_not_relevant() -> None:
    matcher = SpanMatcher(90)
    span = ["laws of the State of Nevada"]
    text = "This Agreement shall be governed by the laws of the State of Nevada."
    assert matcher.chunk_is_relevant(span, text, gold_doc_id="a", chunk_doc_id="a")
    assert not matcher.chunk_is_relevant(span, text, gold_doc_id="a", chunk_doc_id="b")


def test_empty_gold_is_not_a_match() -> None:
    matcher = SpanMatcher(90)
    assert not matcher.spans_in_text([], "anything")
    assert not matcher.spans_in_text(["<omitted>"], "anything")


def test_threshold_rejected() -> None:
    with pytest.raises(ValueError, match="threshold"):
        SpanMatcher(threshold=140)
