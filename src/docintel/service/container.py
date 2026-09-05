from __future__ import annotations

from pathlib import Path
from typing import Any

from docintel.agent.cache import ChunkCache
from docintel.agent.graders import build_grader
from docintel.agent.runtime import AgentRuntime
from docintel.agent.verifiers import build_verifier
from docintel.config import AppConfig
from docintel.config.loader import find_repo_root
from docintel.llm.factory import StructuredCaller, build_caller
from docintel.llm.prompts import PromptBank, load_prompts
from docintel.retrieval.factory import build_retrieval_pipeline


class LazyPipeline:
    """Open Qdrant only on the corpus_technical path."""

    def __init__(self, config: AppConfig, repo_root: Path) -> None:
        self.config = config
        self.repo_root = repo_root
        self._inner: Any | None = None

    def _get(self) -> Any:
        if self._inner is None:
            self._inner = build_retrieval_pipeline(self.config, repo_root=self.repo_root)
        return self._inner

    def ensure(self) -> None:
        self._get()

    def search(self, query: Any) -> Any:
        return self._get().search(query)

    @property
    def reranker(self) -> Any:
        return self._get().reranker

    def close(self) -> None:
        if self._inner is not None:
            self._inner.close()
            self._inner = None


class Container:
    def __init__(
        self,
        config: AppConfig,
        *,
        repo_root: Path | None = None,
        caller: StructuredCaller | None = None,
        pipeline: Any | None = None,
        prompts: PromptBank | None = None,
    ) -> None:
        self.config = config
        self.repo_root = repo_root or find_repo_root()
        self.prompts = prompts or load_prompts()
        self.caller = caller if caller is not None else build_caller(config)
        self.pipeline = pipeline if pipeline is not None else LazyPipeline(config, self.repo_root)
        self.grader = build_grader(
            config.agent.grader.name, self.caller, self.prompts, **config.agent.grader.params
        )
        self.verifier = build_verifier(
            config.agent.verifier.name,
            self.caller,
            self.prompts,
            **config.agent.verifier.params,
        )
        self._mlflow_on = False

    def runtime(self, cache: ChunkCache, query_id: str, started: float) -> AgentRuntime:
        return AgentRuntime(
            config=self.config,
            caller=self.caller,
            prompts=self.prompts,
            cache=cache,
            grader=self.grader,
            verifier=self.verifier,
            pipeline=self.pipeline,
            started=started,
            query_id=query_id,
        )

    def bootstrap_mlflow(self) -> None:
        cfg = self.config.tracking.mlflow
        if self._mlflow_on or not cfg.enabled or not cfg.log_traces:
            return
        try:
            import mlflow
            import mlflow.langchain

            from docintel.evaluation.tracking import resolve_tracking_uri

            mlflow.set_tracking_uri(resolve_tracking_uri(cfg.tracking_uri, self.repo_root))
            mlflow.set_experiment(cfg.experiment)
            mlflow.langchain.autolog()
        except Exception:
            return
        self._mlflow_on = True

    def close(self) -> None:
        closer = getattr(self.pipeline, "close", None)
        if closer:
            closer()
