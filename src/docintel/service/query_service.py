from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from docintel.agent.cache import ChunkCache
from docintel.agent.graph import GraphTick, iter_graph
from docintel.config import config_hash
from docintel.config.schema import TraceSink
from docintel.core.types import Answer, QueryLog, Timing
from docintel.service.container import Container


class QueryService:
    def __init__(self, container: Container) -> None:
        self.container = container
        self.logs: list[QueryLog] = []

    def ask(
        self,
        question: str,
        *,
        session_id: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> tuple[Answer, QueryLog]:
        answer: Answer | None = None
        log: QueryLog | None = None
        for tick in self.ask_iter(question, session_id=session_id, history=history):
            if tick.kind == "done" and tick.state is not None:
                answer = tick.state["answer"]
                log = tick.state["log"]
        if answer is None or log is None:
            raise RuntimeError("ask_iter finished without a result")
        return answer, log

    def ask_iter(
        self,
        question: str,
        *,
        session_id: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[GraphTick]:
        """Yield graph ticks, then a `done` tick whose `state` holds answer+log."""
        cfg = self.container.config
        if TraceSink.MLFLOW in cfg.tracing.sinks:
            self.container.bootstrap_mlflow()
        query_id = str(uuid.uuid4())
        # Nomic + Qdrant open stay off the 90s budget. A first ask after boot or
        # ingest otherwise dies in retrieve_hybrid while the embedder is loading.
        ensure = getattr(self.container.pipeline, "ensure", None)
        if callable(ensure):
            ensure()
        started = time.monotonic()
        cache = ChunkCache()
        runtime = self.container.runtime(cache, query_id, started)
        final = None
        for tick in iter_graph(runtime, question, history=history):
            if tick.kind == "done":
                final = tick.state
                continue
            yield tick
        if final is None:
            raise RuntimeError("graph finished without a values snapshot")
        answer, log = self._finish(final, cache, query_id, question, session_id, started)
        self.logs.append(log)
        yield GraphTick(kind="done", state={"answer": answer, "log": log, "graph": final})

    def close(self) -> None:
        self.container.close()

    def _finish(
        self,
        state: dict[str, Any],
        cache: ChunkCache,
        query_id: str,
        question: str,
        session_id: str | None,
        started: float,
    ) -> tuple[Answer, QueryLog]:
        cfg = self.container.config
        answer = Answer.model_validate(
            state.get("answer") or {"text": "", "route": "corpus_technical"}
        )
        answer.trace_id = query_id
        answer.timings = [Timing.model_validate(t) for t in (state.get("timings") or [])]
        latency_ms = int((time.monotonic() - started) * 1000)
        ctx_ids = list(
            state.get("relevant_ids") or state.get("ranked_ids") or state.get("candidate_ids") or []
        )
        contexts = [row.chunk.text for row in cache.get_many(ctx_ids)]
        trace_path = None
        if TraceSink.JSONL in cfg.tracing.sinks:
            trace_dir = self.container.repo_root / cfg.tracing.jsonl_dir
            raw_trace = state.get("trace") or []
            trace_path = _write_trace(trace_dir, query_id, list(raw_trace))
        log = QueryLog(
            query_id=query_id,
            session_id=session_id,
            question=question,
            route=answer.route,
            config_hash=config_hash(cfg),
            profile=cfg.profile,
            retrieved_chunk_ids=list(state.get("ranked_ids") or state.get("candidate_ids") or []),
            cited_chunk_ids=[c.chunk_id for c in answer.citations],
            cited_doc_ids=list(dict.fromkeys(c.doc_id for c in answer.citations if c.doc_id)),
            answer=answer.text,
            abstained=answer.abstained,
            groundedness=answer.groundedness,
            rewrites=int(state.get("rewrites") or 0),
            latency_ms=latency_ms,
            llm_calls=_llm_calls(answer.timings),
            retrieved_contexts=contexts,
            trace_path=str(trace_path) if trace_path else None,
        )
        return answer, log


_LLM_NODES = frozenset(
    {
        "classify_query",
        "grade_documents",
        "rewrite_query",
        "generate",
        "verify_groundedness",
        "answer_general",
        "clarify",
        "refuse",
    }
)


def _llm_calls(timings: list[Timing]) -> int:
    # ponytail: one call per LLM node. Undercounts `llm_per_chunk` grading (n calls) and
    # structured-output repair retries; upgrade path is a counter on LangChainCaller.
    return sum(1 for t in timings if t.node in _LLM_NODES and t.t_ms > 0)


def _write_trace(root: Path, query_id: str, events: list[dict[str, object]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{query_id}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event, ensure_ascii=True) + "\n")
    return path
