from __future__ import annotations

from docintel.agent.edges import after_classify, after_grade, after_verify


def test_route_tables() -> None:
    assert after_classify({"route": "general"}) == "answer_general"
    assert after_classify({"route": "ambiguous"}) == "clarify"
    assert after_classify({"route": "out_of_scope"}) == "refuse"
    assert after_classify({"route": "corpus_technical"}) == "plan_retrieval"
    assert after_grade({"relevant_ids": ["a", "b"]}, min_relevant=2, max_rewrites=2) == "generate"
    assert after_grade({"relevant_ids": [], "rewrites": 0}, min_relevant=2, max_rewrites=2) == (
        "rewrite_query"
    )
    assert after_grade({"relevant_ids": [], "rewrites": 2}, min_relevant=2, max_rewrites=2) == (
        "abstain"
    )
    # one relevant chunk after the rewrite cap still answers; never abstain with evidence
    assert after_grade({"relevant_ids": ["a"], "rewrites": 2}, min_relevant=2, max_rewrites=2) == (
        "generate"
    )
    assert after_verify({"grounded": True, "regen_count": 0}) == "finalize"
    assert after_verify({"grounded": False, "regen_count": 0}) == "regenerate_strict"
    assert after_verify({"grounded": False, "regen_count": 1}) == "abstain"
