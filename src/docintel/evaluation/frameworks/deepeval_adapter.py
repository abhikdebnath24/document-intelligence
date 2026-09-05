from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel

from docintel.config import AppConfig
from docintel.evaluation.frameworks.base import BaseGenerationEvaluator, EvalResult, EvalSample
from docintel.llm.factory import build_chat_model
from docintel.llm.structured import structured


class LangchainJudge:
    """DeepEvalBaseLLM wrapper. `schema` must return a Pydantic instance when set."""

    def __init__(self, model: Any, name: str) -> None:
        self.model = model
        self.model_name = name

    def load_model(self) -> Any:
        return self.model

    def get_model_name(self) -> str:
        return self.model_name

    def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
        if schema is None:
            raw = self.model.invoke(prompt)
            return getattr(raw, "content", raw)
        return structured(self.model, schema, prompt)

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
        return self.generate(prompt, schema)


# metric -> LLMTestCase fields it needs beyond input/actual_output
_NEEDS: dict[str, tuple[str, ...]] = {
    "faithfulness": ("retrieval_context",),
    "answer_relevancy": (),
    "contextual_precision": ("retrieval_context", "expected_output"),
    "contextual_recall": ("retrieval_context", "expected_output"),
    "contextual_relevancy": ("retrieval_context",),
    "geval_clause_citation": ("retrieval_context",),
}


def _import_deepeval() -> None:
    # local judge only; no Confident AI telemetry / login prompt
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    os.environ.setdefault("DEEPEVAL_DISABLE_PROGRESS_BAR", "YES")


def _wrap_judge(config: AppConfig) -> Any:
    _import_deepeval()
    from deepeval.models import DeepEvalBaseLLM

    class _Judge(LangchainJudge, DeepEvalBaseLLM):  # type: ignore[no-untyped-call]
        def __init__(self, model: Any, name: str) -> None:
            LangchainJudge.__init__(self, model, name)
            self.name = name  # DeepEvalBaseLLM.__init__ is skipped; it sets this

    spec = config.llm.roles[config.evaluation.judge_role]
    model = build_chat_model(config.evaluation.judge_role, config)
    return _Judge(model, spec.model)


class DeepEvalEvaluator(BaseGenerationEvaluator):
    def __init__(
        self,
        config: AppConfig,
        *,
        measure: Any | None = None,
        judge: Any | None = None,
    ) -> None:
        self.config = config
        self._measure = measure
        self._judge = judge

    def evaluate(self, samples: list[EvalSample]) -> EvalResult:
        judge = self._judge or _wrap_judge(self.config)
        metrics = self._measure or _build_metrics(self.config, judge)
        per_sample: list[dict[str, Any]] = []
        skipped: list[str] = []
        notes: list[str] = []
        sums: dict[str, list[float]] = {name: [] for name in metrics}
        for sample in samples:
            row: dict[str, Any] = {"id": sample.id}
            if sample.expected_abstain or sample.abstained:
                skipped.append(sample.id)
                row["skipped"] = True
                per_sample.append(row)
                continue
            case = _test_case(sample)
            for name, metric in metrics.items():
                if any(not getattr(case, field) for field in _NEEDS.get(name, ())):
                    notes.append(f"{sample.id}:{name}:missing_input")
                    continue
                try:
                    metric.measure(case)
                    score = float(getattr(metric, "score", 0.0) or 0.0)
                except Exception as exc:
                    notes.append(f"{sample.id}:{name}:{type(exc).__name__}")
                    continue
                row[name] = score
                sums[name].append(score)
            per_sample.append(row)
        overall = {name: (sum(vals) / len(vals) if vals else 0.0) for name, vals in sums.items()}
        if skipped:
            notes.append("abstain rows excluded from DeepEval averages")
        return EvalResult(
            framework="deepeval",
            overall=overall,
            per_sample=per_sample,
            skipped_ids=skipped,
            notes=notes,
        )


def _test_case(sample: EvalSample) -> Any:
    _import_deepeval()
    from deepeval.test_case import LLMTestCase

    # `context` is the gold context slot; we have no gold chunks, only gold spans
    contexts: list[Any] | None = list(sample.retrieved_contexts) or None
    return LLMTestCase(
        input=sample.question,
        actual_output=sample.answer,
        expected_output=sample.reference,
        retrieval_context=contexts,
    )


def _build_metrics(config: AppConfig, judge: Any) -> dict[str, Any]:
    _import_deepeval()
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
    )

    threshold = config.evaluation.deepeval.threshold
    wanted = set(config.evaluation.deepeval.metrics)
    catalog: dict[str, Any] = {
        "faithfulness": FaithfulnessMetric(model=judge, threshold=threshold, async_mode=False),
        "answer_relevancy": AnswerRelevancyMetric(
            model=judge, threshold=threshold, async_mode=False
        ),
        "contextual_precision": ContextualPrecisionMetric(
            model=judge, threshold=threshold, async_mode=False
        ),
        "contextual_recall": ContextualRecallMetric(
            model=judge, threshold=threshold, async_mode=False
        ),
        "contextual_relevancy": ContextualRelevancyMetric(
            model=judge, threshold=threshold, async_mode=False
        ),
    }
    out = {name: metric for name, metric in catalog.items() if name in wanted}
    rubric = (config.evaluation.deepeval.geval_rubrics or ["answer cites the governing clause"])[0]
    out["geval_clause_citation"] = _geval(judge, rubric, threshold)
    return out


def _geval(judge: Any, rubric: str, threshold: float) -> Any:
    from deepeval.metrics import GEval

    params = _geval_params()
    return GEval(
        name="ClauseCitation",
        criteria=rubric,
        evaluation_params=params,
        model=judge,
        threshold=threshold,
        async_mode=False,
    )


def _geval_params() -> list[Any]:
    import deepeval.test_case as tc

    # SingleTurnParams in deepeval >= 4; LLMTestCaseParams in 3.x
    params: Any = getattr(tc, "SingleTurnParams", None) or getattr(tc, "LLMTestCaseParams")
    names = ("INPUT", "ACTUAL_OUTPUT", "RETRIEVAL_CONTEXT")
    return [getattr(params, name) for name in names if hasattr(params, name)]
