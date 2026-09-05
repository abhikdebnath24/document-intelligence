from __future__ import annotations

from pathlib import Path

import pytest

from docintel.agent.cache import ChunkCache
from docintel.agent.graph import iter_graph, run_graph, step_label
from docintel.agent.runtime import AgentRuntime
from docintel.config.loader import load_config
from docintel.config.schema import TraceSink
from docintel.core.types import Chunk, RetrievedChunk
from docintel.llm.factory import ScriptedCaller
from docintel.llm.prompts import load_prompts
from docintel.llm.schemas import (
    CiteDraft,
    ClarifyOut,
    GenerateOut,
    GradeBatchOut,
    GradeItemOut,
    RewriteOut,
    RouteOut,
    VerifyOut,
)
from docintel.retrieval.rerankers import NoOpReranker
from docintel.service.container import Container
from docintel.service.query_service import QueryService

ROOT = Path(__file__).resolve().parents[2]
ATTACK = (ROOT / "tests" / "fixtures" / "adversarial_chunk.txt").read_text().strip()


class FakePipeline:
    """Returns rows per call in order; last list repeats. Records filters it received."""

    def __init__(self, rows: list[RetrievedChunk], *more: list[RetrievedChunk]) -> None:
        self.batches = [rows, *more]
        self.calls = 0
        self.filters: list[dict[str, object]] = []
        self.reranker = NoOpReranker()

    def search(self, query: object) -> list[RetrievedChunk]:
        self.filters.append(dict(getattr(query, "filters", {})))
        batch = self.batches[min(self.calls, len(self.batches) - 1)]
        self.calls += 1
        return list(batch)


def _row(cid: str, text: str, doc_id: str = "acme") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=cid, doc_id=doc_id, text=text, page_start=4, page_end=4, chunk_idx=0),
        score=1.0,
        source="fused",
        rank=1,
    )


def _cfg(**agent: object):
    cfg = load_config("dev_cpu", repo_root=ROOT)
    data = cfg.agent.model_dump()
    data.update(agent)
    return cfg.model_copy(update={"agent": cfg.agent.model_validate(data)})


def _rt(
    cfg,
    caller,
    rows: list[RetrievedChunk] | FakePipeline,
    query_id: str = "q-test",
) -> AgentRuntime:
    from docintel.agent.graders import build_grader
    from docintel.agent.verifiers import build_verifier

    prompts = load_prompts()
    pipeline = rows if isinstance(rows, FakePipeline) else FakePipeline(rows)
    return AgentRuntime(
        config=cfg,
        caller=caller,
        prompts=prompts,
        cache=ChunkCache(),
        grader=build_grader(cfg.agent.grader.name, caller, prompts, **cfg.agent.grader.params),
        verifier=build_verifier(
            cfg.agent.verifier.name, caller, prompts, **cfg.agent.verifier.params
        ),
        pipeline=pipeline,
        started=__import__("time").monotonic(),
        query_id=query_id,
    )


def test_cited_success_ignores_adversarial_chunk() -> None:
    cfg = _cfg(min_relevant_chunks=1, max_rewrites=0)
    real = _row("c-real", "This Agreement shall be governed by the laws of the State of Delaware.")
    atk = _row("c-atk", ATTACK)
    caller = ScriptedCaller(
        {
            "router": [RouteOut(route="corpus_technical", reason="named clause")],
            "grader": [
                GradeBatchOut(
                    grades=[
                        GradeItemOut(chunk_id="c-real", relevant=True, reason="governing law"),
                        GradeItemOut(chunk_id="c-atk", relevant=False, reason="injection"),
                    ]
                )
            ],
            "generation": [
                GenerateOut(
                    answer="Delaware law governs.",
                    citations=[CiteDraft(chunk_id="c-real", quote="laws of the State of Delaware")],
                )
            ],
            "verifier": [VerifyOut(grounded=True, score=1.0, unsupported_claims=[])],
        }
    )
    state = run_graph(_rt(cfg, caller, [real, atk], "trace-success"), "Which law governs Acme?")
    answer = state["answer"]
    assert answer["abstained"] is False
    assert "Delaware" in answer["text"]
    assert "Nevada" not in answer["text"]
    assert [c["chunk_id"] for c in answer["citations"]] == ["c-real"]
    nodes = [e["node"] for e in state["trace"]]
    assert nodes[:5] == [
        "classify_query",
        "plan_retrieval",
        "retrieve_hybrid",
        "rerank",
        "grade_documents",
    ]
    assert "generate" in nodes and "finalize" in nodes


def test_abstain_when_nothing_relevant() -> None:
    cfg = _cfg(min_relevant_chunks=2, max_rewrites=0)
    caller = ScriptedCaller(
        {
            "router": [RouteOut(route="corpus_technical", reason="clause")],
            "grader": [
                GradeBatchOut(grades=[GradeItemOut(chunk_id="c1", relevant=False, reason="no")])
            ],
        }
    )
    state = run_graph(
        _rt(cfg, caller, [_row("c1", "unrelated")], "trace-abstain"), "Any most-favored nation?"
    )
    assert state["answer"]["abstained"] is True
    assert "does not contain" in state["answer"]["text"]
    assert "abstain" in [e["node"] for e in state["trace"]]


def test_general_and_refuse_and_clarify() -> None:
    cfg = _cfg()
    general = ScriptedCaller(
        {
            "router": [RouteOut(route="general", reason="definition")],
            "generation": ["Indemnity shifts loss between parties."],
        }
    )
    g = run_graph(_rt(cfg, general, []), "What is an indemnity clause?")
    assert g["route"] == "general"
    assert "general knowledge" in g["answer"]["text"]
    assert g["answer"]["citations"] == []

    refuse = ScriptedCaller({"router": [RouteOut(route="out_of_scope", reason="weather")]})
    r = run_graph(_rt(cfg, refuse, []), "What is the weather in Paris?")
    assert r["route"] == "out_of_scope"
    assert "outside the scope" in r["answer"]["text"]

    clarify = ScriptedCaller(
        {
            "router": [
                RouteOut(route="ambiguous", reason="which contract"),
                ClarifyOut(question="Which distributor agreement?"),
            ]
        }
    )
    c = run_graph(_rt(cfg, clarify, []), "What is the termination notice?")
    assert c["route"] == "ambiguous"
    assert "Which distributor" in c["answer"]["text"]


def test_rewrite_loop_then_abstain() -> None:
    cfg = _cfg(min_relevant_chunks=2, max_rewrites=1)
    caller = ScriptedCaller(
        {
            "router": [
                RouteOut(route="corpus_technical", reason="clause"),
                RewriteOut(query="most favored nation clause MFN", reason="synonym"),
            ],
            "grader": [
                GradeBatchOut(grades=[GradeItemOut(chunk_id="c1", relevant=False, reason="no")]),
                GradeBatchOut(
                    grades=[GradeItemOut(chunk_id="c1", relevant=False, reason="still no")]
                ),
            ],
        }
    )
    state = run_graph(_rt(cfg, caller, [_row("c1", "unrelated")]), "Any MFN?")
    assert state["rewrites"] == 1
    assert state["answer"]["abstained"] is True


def test_rewrite_keeps_first_pass_evidence_and_router_hints_stay_out_of_filters() -> None:
    # pass 1 finds one relevant chunk (< min 2); rewrite finds nothing new. The first
    # chunk must survive the rewrite and be answered, not dropped into an abstain.
    cfg = _cfg(min_relevant_chunks=2, max_rewrites=1)
    real = _row("c-real", "This Agreement shall be governed by the laws of the State of Delaware.")
    pipeline = FakePipeline([real], [])
    caller = ScriptedCaller(
        {
            "router": [
                RouteOut(
                    route="corpus_technical",
                    reason="x",
                    agreement_type="License",
                    doc_hint="Acme",
                ),
                RewriteOut(query="governing law Acme", reason="synonym"),
            ],
            "grader": [
                GradeBatchOut(
                    grades=[GradeItemOut(chunk_id="c-real", relevant=True, reason="law")]
                ),
                GradeBatchOut(
                    grades=[GradeItemOut(chunk_id="c-real", relevant=True, reason="law")]
                ),
            ],
            "generation": [
                GenerateOut(
                    answer="Delaware law governs.",
                    citations=[CiteDraft(chunk_id="c-real", quote="laws of the State of Delaware")],
                )
            ],
            "verifier": [VerifyOut(grounded=True, score=1.0, unsupported_claims=[])],
        }
    )
    state = run_graph(_rt(cfg, caller, pipeline), "Which law governs Acme?")
    assert state["rewrites"] == 1
    assert state["answer"]["abstained"] is False
    assert [c["chunk_id"] for c in state["answer"]["citations"]] == ["c-real"]
    assert state["doc_hint"] == "Acme"
    # router output never becomes a Qdrant payload filter
    assert all("doc_hint" not in f and "agreement_type" not in f for f in pipeline.filters)


def test_malformed_generation_raises() -> None:
    cfg = _cfg(min_relevant_chunks=1, max_rewrites=0)
    caller = ScriptedCaller(
        {
            "router": [RouteOut(route="corpus_technical", reason="x")],
            "grader": [
                GradeBatchOut(grades=[GradeItemOut(chunk_id="c1", relevant=True, reason="yes")])
            ],
            "generation": ["<<<not-json>>>"],
        }
    )
    with pytest.raises(Exception):
        run_graph(_rt(cfg, caller, [_row("c1", "laws of Delaware")]), "Which law?")


def test_query_service_writes_trace(tmp_path: Path) -> None:
    cfg = _cfg(min_relevant_chunks=1, max_rewrites=0)
    cfg = cfg.model_copy(
        update={
            "tracing": cfg.tracing.model_copy(
                update={"jsonl_dir": "traces", "sinks": [TraceSink.JSONL]}
            )
        }
    )
    caller = ScriptedCaller({"router": [RouteOut(route="out_of_scope", reason="no")]})
    container = Container(
        cfg,
        repo_root=tmp_path,
        caller=caller,
        pipeline=FakePipeline([]),
        prompts=load_prompts(),
    )
    svc = QueryService(container)
    answer, log = svc.ask("who won the world cup?")
    svc.close()
    assert answer.route == "out_of_scope"
    assert log.trace_path
    path = Path(log.trace_path)
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "classify_query" in text
    assert "finalize" in text


def test_iter_graph_emits_start_before_done() -> None:
    cfg = _cfg()
    caller = ScriptedCaller(
        {
            "router": [RouteOut(route="general", reason="definition")],
            "generation": ["Indemnity shifts loss between parties."],
        }
    )
    starts: list[str] = []
    final = None
    for tick in iter_graph(_rt(cfg, caller, []), "What is an indemnity clause?"):
        if tick.kind == "start":
            starts.append(tick.node)
        elif tick.kind == "done":
            final = tick.state
    assert starts[0] == "classify_query"
    assert "answer_general" in starts
    assert "finalize" in starts
    assert final is not None
    assert final["route"] == "general"
    assert step_label("retrieve_hybrid") == "Searching dense + BM25"


def test_ask_iter_yields_done_answer() -> None:
    cfg = _cfg()
    caller = ScriptedCaller({"router": [RouteOut(route="out_of_scope", reason="no")]})
    container = Container(
        cfg,
        repo_root=ROOT,
        caller=caller,
        pipeline=FakePipeline([]),
        prompts=load_prompts(),
    )
    svc = QueryService(container)
    kinds = []
    answer = None
    for tick in svc.ask_iter("who won?"):
        kinds.append(tick.kind)
        if tick.kind == "done" and tick.state is not None:
            answer = tick.state["answer"]
    svc.close()
    assert "start" in kinds
    assert kinds[-1] == "done"
    assert answer is not None
    assert answer.route == "out_of_scope"
