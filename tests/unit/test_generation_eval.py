from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from docintel.config import load_config
from docintel.evaluation.custom_metrics import (
    agreement_markdown,
    custom_metrics,
    framework_agreement,
    tag_miss,
    top_misses,
)
from docintel.evaluation.experiment import FinalistGateError
from docintel.evaluation.frameworks.base import EvalResult, EvalSample, samples_from_records
from docintel.evaluation.frameworks.deepeval_adapter import LangchainJudge
from docintel.evaluation.frameworks.ragas_adapter import RagasEvaluator
from docintel.evaluation.generation_eval import run_generation_eval
from docintel.evaluation.gold import QAItem
from docintel.evaluation.tracking import mlflow_run

ROOT = Path(__file__).resolve().parents[2]


def _item(**kwargs: object) -> QAItem:
    payload = {
        "id": "q_1",
        "doc_stem": "acme_1",
        "agreement_type": "License",
        "category": "Governing Law",
        "bucket": "slot",
        "question": "Which law governs?",
        "gold_spans": ["New York"],
        "gold_answer": "New York",
        "expected_route": "corpus_technical",
        "expected_abstain": False,
        "split": "dev",
    }
    payload.update(kwargs)
    return QAItem.model_validate(payload)


def test_samples_from_records_citation_and_skip_unknown() -> None:
    items = [_item()]
    records = [
        {
            "id": "q_1",
            "answer": "New York law.",
            "route": "corpus_technical",
            "abstained": False,
            "retrieved_contexts": ["This Agreement is governed by New York law."],
            "citations": [{"quote": "governed by New York law"}],
            "latency_ms": 10,
            "llm_calls": 4,
            "groundedness": 0.9,
        },
        {"id": "missing"},
    ]
    samples = samples_from_records(items, records)
    assert len(samples) == 1
    assert samples[0].reference == "New York"
    assert samples[0].citations_valid is True


def test_custom_metrics_route_and_abstention() -> None:
    samples = [
        EvalSample(
            id="a",
            question="q",
            answer="yes",
            expected_route="corpus_technical",
            predicted_route="corpus_technical",
            expected_abstain=True,
            abstained=True,
            citations_valid=True,
            latency_ms=100,
            llm_calls=2,
            groundedness=0.0,
            bucket="no_answer",
        ),
        EvalSample(
            id="b",
            question="q",
            answer="no",
            expected_route="general",
            predicted_route="corpus_technical",
            expected_abstain=False,
            abstained=True,
            citations_valid=False,
            latency_ms=200,
            llm_calls=4,
            groundedness=0.5,
            bucket="slot",
        ),
    ]
    got = custom_metrics(samples)
    assert got["n"] == 2
    assert got["route_accuracy"] == 0.5
    assert got["abstention_precision"] == 0.5
    assert got["abstention_recall"] == 1.0
    assert got["citation_validity"] == 0.5
    assert got["latency_p50_ms"] == 100
    assert got["llm_calls_per_query"] == 3.0


def test_framework_agreement_spearman_and_markdown() -> None:
    ragas = EvalResult(
        framework="ragas",
        per_sample=[
            {"id": "q1", "faithfulness": 1.0},
            {"id": "q2", "faithfulness": 0.2},
            {"id": "q3", "faithfulness": 0.8},
        ],
    )
    deepeval = EvalResult(
        framework="deepeval",
        per_sample=[
            {"id": "q1", "faithfulness": 1.0},
            {"id": "q2", "faithfulness": 0.8},
            {"id": "q3", "faithfulness": 0.7},
        ],
    )
    report = framework_agreement(ragas, deepeval)
    assert report["n"] == 3
    assert report["headline"] == "ragas"
    assert report["disagree_gt_0.3"][0]["id"] == "q2"
    md = agreement_markdown(report)
    assert "RAGAS faithfulness is the headline" in md
    assert "q2" in md


def test_tag_miss_and_top_misses() -> None:
    rows = [
        {"id": "a", "doc_id": "gold", "doc_ids": ["other"], "hit@10": 0, "r@10": 0, "mrr": 0},
        {"id": "b", "doc_id": "gold", "doc_ids": ["gold"], "hit@5": 0, "hit@10": 1, "r@10": 0.5},
        {"id": "c", "doc_id": "gold", "doc_ids": ["gold"], "hit@10": 1, "r@10": 1, "mrr": 1},
    ]
    assert tag_miss(rows[0]) == "exact_term_miss"
    assert tag_miss(rows[1]) == "reranker_demoted_gold"
    top = top_misses(rows, n=20)
    assert [r["id"] for r in top] == ["a", "b"]


def test_ragas_adapter_skips_abstain_and_uses_injected_metric() -> None:
    class _Box:
        def __init__(self, value: float) -> None:
            self.value = value

    class _Metric:
        async def ascore(
            self,
            user_input: str,
            response: str,
            retrieved_contexts: list[str],
            reference: str = "",
        ) -> _Box:
            _ = user_input, response, retrieved_contexts, reference
            return _Box(1.0)

    cfg = load_config("dev_cpu", repo_root=ROOT)
    ev = RagasEvaluator(cfg, metrics={"faithfulness": _Metric()})
    result = ev.evaluate(
        [
            EvalSample(id="ok", question="q", answer="a", retrieved_contexts=["c"]),
            EvalSample(id="abs", question="q", answer="no", expected_abstain=True, abstained=True),
        ]
    )
    assert result.overall["faithfulness"] == 1.0
    assert result.skipped_ids == ["abs"]
    assert result.per_sample[0]["faithfulness"] == 1.0


def test_ascore_kwargs_gates_missing_required_inputs() -> None:
    from docintel.evaluation.frameworks.ragas_adapter import _ascore_kwargs

    async def needs_ref(user_input: str, reference: str, retrieved_contexts: list[str]) -> None:
        _ = user_input, reference, retrieved_contexts

    async def faith(user_input: str, response: str, retrieved_contexts: list[str]) -> None:
        _ = user_input, response, retrieved_contexts

    with_ref = EvalSample(id="a", question="q", answer="a", retrieved_contexts=["c"], reference="r")
    no_ref = EvalSample(id="b", question="q", answer="a", retrieved_contexts=["c"])
    general = EvalSample(id="c", question="q", answer="a")
    assert _ascore_kwargs(needs_ref, with_ref) == {
        "user_input": "q",
        "reference": "r",
        "retrieved_contexts": ["c"],
    }
    assert _ascore_kwargs(needs_ref, no_ref) is None
    assert _ascore_kwargs(faith, no_ref) is not None
    assert _ascore_kwargs(faith, general) is None


def test_generation_eval_empty_frameworks_means_custom_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import shutil

    import docintel.evaluation.generation_eval as ge

    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    shutil.copytree(ROOT / "evals", tmp_path / "evals")
    (tmp_path / "pyproject.toml").write_text("")
    cfg = load_config("dev_cpu", repo_root=tmp_path)
    cfg.tracking.mlflow.enabled = False
    monkeypatch.setattr(ge, "_ask_all", lambda *a, **k: [])
    calls: list[str] = []
    monkeypatch.setattr(ge, "_eval_ragas", lambda *a, **k: calls.append("ragas"))
    monkeypatch.setattr(ge, "_eval_deepeval", lambda *a, **k: calls.append("deepeval"))
    out = run_generation_eval(cfg, split="dev", repo_root=tmp_path, frameworks=[])
    assert calls == []
    assert out.is_file()


def test_langchain_judge_honours_schema() -> None:
    class Out(BaseModel):
        ok: bool

    class _Model:
        def invoke(self, prompt: object) -> object:
            class _Raw:
                content = "plain"

            _ = prompt
            return _Raw()

        def with_structured_output(self, schema: type[BaseModel]) -> object:
            class _Bound:
                def invoke(self, messages: object) -> Out:
                    _ = messages
                    return schema(ok=True)

            return _Bound()

    judge = LangchainJudge(_Model(), "fake")
    assert judge.generate("hi") == "plain"
    parsed = judge.generate("hi", schema=Out)
    assert parsed.ok is True


def test_l2_test_split_refuses_unlocked_profile() -> None:
    cfg = load_config("exp_dense_only", repo_root=ROOT)
    with pytest.raises(FinalistGateError, match="not in"):
        run_generation_eval(cfg, split="test", repo_root=ROOT)


def test_mlflow_disabled_yields_none(tmp_path: Path) -> None:
    cfg = load_config("dev_cpu", repo_root=ROOT)
    cfg = cfg.model_copy(update={"tracking": cfg.tracking.model_copy(deep=True)})
    cfg.tracking.mlflow.enabled = False
    with mlflow_run(cfg, repo_root=tmp_path, split="dev", layer="L2") as run_id:
        assert run_id is None
