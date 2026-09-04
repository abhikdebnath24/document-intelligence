from __future__ import annotations

import json
from pathlib import Path

import pytest

from docintel.evaluation.gold import QAItem, assert_document_disjoint, load_qa_set

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "evals" / "qa_dev.json"
TEST = ROOT / "evals" / "qa_test.json"
MANIFEST = ROOT / "data_manifest" / "corpus_manifest.json"


def test_assert_document_disjoint_accepts_empty_stems() -> None:
    dev = [
        QAItem(
            id="q_001",
            doc_stem="a",
            agreement_type="Distributor",
            category="Governing Law",
            bucket="slot",
            question="q",
            split="dev",
        ),
        QAItem(
            id="q_002",
            doc_stem="",
            agreement_type="",
            category="general",
            bucket="general",
            question="what is force majeure?",
            expected_route="general",
            split="dev",
        ),
    ]
    test = [
        QAItem(
            id="q_003",
            doc_stem="b",
            agreement_type="License",
            category="Exclusivity",
            bucket="yes_span",
            question="q",
            split="test",
        ),
        QAItem(
            id="q_004",
            doc_stem="",
            agreement_type="",
            category="out_of_scope",
            bucket="general",
            question="weather in pune?",
            expected_route="out_of_scope",
            split="test",
        ),
    ]
    assert_document_disjoint(dev, test)


def test_assert_document_disjoint_rejects_overlap() -> None:
    a = QAItem(
        id="q_001",
        doc_stem="same",
        agreement_type="License",
        category="Exclusivity",
        bucket="yes_span",
        question="q",
        split="dev",
    )
    b = a.model_copy(update={"id": "q_002", "split": "test"})
    with pytest.raises(ValueError, match="both splits"):
        assert_document_disjoint([a], [b])


def test_committed_eval_sets() -> None:
    if not DEV.is_file() or not TEST.is_file() or not MANIFEST.is_file():
        pytest.skip("eval artifacts not generated")
    dev = load_qa_set(DEV)
    test = load_qa_set(TEST)
    assert_document_disjoint(dev, test)
    union = dev + test
    assert len(test) >= 20
    assert {q.bucket for q in union} == {"slot", "yes_span", "no_answer", "cross_ref", "general"}
    assert len({q.id for q in union}) == len(union)
    for q in union:
        if q.bucket == "general":
            assert not q.doc_stem and not q.gold_spans
        elif q.bucket == "no_answer":
            assert q.expected_abstain and not q.gold_spans
        else:
            assert q.gold_spans and not q.expected_abstain
    contract = [q for q in union if q.doc_stem]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    docs = manifest["documents"]
    eval_docs = [d for d in docs if d["split"] == "index_and_eval"]
    eval_stems = {d["doc_stem"] for d in eval_docs}
    qa_stems = {q.doc_stem for q in contract}
    assert qa_stems <= eval_stems
    missing = eval_stems - qa_stems
    assert not missing, f"eval docs without a question: {sorted(missing)[:8]}"
    assert len({d["sha256"] for d in docs}) == len(docs), "duplicate PDF bytes in manifest"
    group_of = {d["doc_stem"]: d["group"] for d in eval_docs}
    for split_items in (dev, test):
        groups = {group_of[q.doc_stem] for q in split_items if q.doc_stem}
        assert groups == {"core", "ip", "other"}
