from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Callable
from typing import Any

from docintel.config import AppConfig
from docintel.evaluation.frameworks.base import BaseGenerationEvaluator, EvalResult, EvalSample
from docintel.llm.factory import parse_model_ref, require_provider_keys


def _metric_value(result: Any) -> float:
    """`MetricResult.value` in 0.4; bare number from older/test doubles."""
    if result is None:
        return 0.0
    if isinstance(result, int | float):
        return float(result)
    value = getattr(result, "value", None)
    return float(value) if isinstance(value, int | float) else 0.0


def _ascore_kwargs(ascore: Callable[..., Any], sample: EvalSample) -> dict[str, Any] | None:
    """Kwargs for this metric's `ascore`, or None when a required input is missing
    (no gold reference, no retrieved contexts). Passing "" would score garbage."""
    params = inspect.signature(ascore).parameters
    payload: dict[str, Any] = {
        "user_input": sample.question,
        "response": sample.answer,
        "retrieved_contexts": sample.retrieved_contexts,
        "reference": sample.reference,
    }
    out: dict[str, Any] = {}
    for name, param in params.items():
        if name not in payload:
            continue
        value = payload[name]
        if value:
            out[name] = value
        elif param.default is inspect.Parameter.empty:
            return None
    return out


class RagasEvaluator(BaseGenerationEvaluator):
    """RAGAS 0.4 collections API. Per-sample `ascore()`; skip abstain from faithfulness."""

    def __init__(
        self,
        config: AppConfig,
        *,
        metrics: dict[str, Any] | None = None,
        runner: Callable[[Any], Any] | None = None,
    ) -> None:
        self.config = config
        self._metrics = metrics
        self._runner = runner or (lambda coro: asyncio.run(coro))

    def evaluate(self, samples: list[EvalSample]) -> EvalResult:
        metrics = self._metrics or _build_metrics(self.config)
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
            for name, metric in metrics.items():
                kwargs = _ascore_kwargs(metric.ascore, sample)
                if kwargs is None:
                    notes.append(f"{sample.id}:{name}:missing_input")
                    continue
                try:
                    score = _metric_value(self._runner(metric.ascore(**kwargs)))
                except Exception as exc:
                    notes.append(f"{sample.id}:{name}:{type(exc).__name__}")
                    continue
                row[name] = score
                sums[name].append(score)
            per_sample.append(row)
        overall = {name: (sum(vals) / len(vals) if vals else 0.0) for name, vals in sums.items()}
        if skipped:
            notes.append("abstain rows excluded from RAGAS averages")
        return EvalResult(
            framework="ragas",
            overall=overall,
            per_sample=per_sample,
            skipped_ids=skipped,
            notes=notes,
        )


_RAGAS_PROVIDER = {"anthropic": "anthropic", "openai": "openai", "google_genai": "google"}


def _shim_langchain_community() -> None:
    """ragas 0.4.3 imports `langchain_community.chat_models.vertexai`, removed in
    langchain-community 0.4 (the series langchain 1.x needs). ragas only uses the
    class in an `isinstance` list, so an empty stand-in is safe."""
    import importlib
    import sys
    import types

    name = "langchain_community.chat_models.vertexai"
    try:
        importlib.import_module(name)
        return
    except ImportError:
        pass
    mod = types.ModuleType(name)

    class ChatVertexAI:  # pragma: no cover - never instantiated
        pass

    mod.ChatVertexAI = ChatVertexAI  # type: ignore[attr-defined]
    sys.modules[name] = mod


def _build_metrics(config: AppConfig) -> dict[str, Any]:
    _shim_langchain_community()
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
        NoiseSensitivity,
    )

    require_provider_keys(config)
    spec = config.llm.roles[config.evaluation.judge_role]
    provider, model_id = parse_model_ref(spec.model, config.llm.default_provider)
    model_name = model_id.split(":", 1)[-1]
    # provider is NOT auto-detected from the client in 0.4.3; default is "openai"
    llm = llm_factory(
        model_name, provider=_RAGAS_PROVIDER[provider], client=_async_client(provider)
    )
    wanted = set(config.evaluation.ragas.metrics)
    catalog: dict[str, Any] = {
        "faithfulness": Faithfulness(llm=llm),
        "context_precision_with_reference": ContextPrecisionWithReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "noise_sensitivity": NoiseSensitivity(llm=llm),
    }
    # answer_relevancy needs an embeddings client too; wire it when an embedder is configured
    return {name: metric for name, metric in catalog.items() if name in wanted}


def _async_client(provider: str) -> Any:
    if provider == "anthropic":
        from anthropic import AsyncAnthropic

        return AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    if provider == "openai":
        from openai import AsyncOpenAI

        return AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if provider == "google_genai":
        from google import genai

        return genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    raise ValueError(f"unsupported ragas judge provider {provider!r}")
