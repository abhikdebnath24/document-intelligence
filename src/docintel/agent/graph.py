from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict

from docintel.agent.nodes import Nodes
from docintel.agent.runtime import AgentRuntime
from docintel.agent.state import RAGState

# LangGraph also emits these bookkeeping names; the UI should skip them.
_SKIP_NODES = frozenset({START, END, "__start__", "__end__"})


class GraphTick(BaseModel):
    """One graph event for a live UI. `start` fires before the node body."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["start", "end", "done"]
    node: str = ""
    state: Any = None


STEP_LABELS: dict[str, str] = {
    "classify_query": "Classifying the question",
    "plan_retrieval": "Planning retrieval filters",
    "retrieve_hybrid": "Searching dense + BM25",
    "rerank": "Reranking candidates",
    "grade_documents": "Grading retrieved chunks",
    "rewrite_query": "Rewriting the query",
    "generate": "Drafting the answer",
    "verify_groundedness": "Checking claims against evidence",
    "regenerate_strict": "Rewriting unsupported claims",
    "abstain": "No supporting clause found",
    "answer_general": "Answering from general knowledge",
    "clarify": "Need a more specific question",
    "refuse": "Refusing an out-of-scope question",
    "finalize": "Packaging citations",
}


def step_label(node: str) -> str:
    return STEP_LABELS.get(node, node.replace("_", " "))


def build_graph(nodes: Nodes) -> Any:
    """Compile the Adaptive/Corrective/Self-RAG graph (LangGraph StateGraph)."""
    builder = StateGraph(RAGState)
    builder.add_node("classify_query", nodes.classify_query)
    builder.add_node("plan_retrieval", nodes.plan_retrieval)
    builder.add_node("retrieve_hybrid", nodes.retrieve_hybrid)
    builder.add_node("rerank", nodes.rerank)
    builder.add_node("grade_documents", nodes.grade_documents)
    builder.add_node("rewrite_query", nodes.rewrite_query)
    builder.add_node("generate", nodes.generate)
    builder.add_node("verify_groundedness", nodes.verify_groundedness)
    builder.add_node("regenerate_strict", nodes.regenerate_strict)
    builder.add_node("abstain", nodes.abstain)
    builder.add_node("answer_general", nodes.answer_general)
    builder.add_node("clarify", nodes.clarify)
    builder.add_node("refuse", nodes.refuse)
    builder.add_node("finalize", nodes.finalize)

    builder.add_edge(START, "classify_query")
    builder.add_conditional_edges(
        "classify_query",
        nodes.route_after_classify,
        {
            "answer_general": "answer_general",
            "clarify": "clarify",
            "refuse": "refuse",
            "plan_retrieval": "plan_retrieval",
        },
    )
    builder.add_edge("plan_retrieval", "retrieve_hybrid")
    builder.add_edge("retrieve_hybrid", "rerank")
    builder.add_edge("rerank", "grade_documents")
    builder.add_conditional_edges(
        "grade_documents",
        nodes.route_after_grade,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "abstain": "abstain",
        },
    )
    builder.add_edge("rewrite_query", "retrieve_hybrid")
    builder.add_edge("generate", "verify_groundedness")
    builder.add_edge("regenerate_strict", "verify_groundedness")
    builder.add_conditional_edges(
        "verify_groundedness",
        nodes.route_after_verify,
        {
            "finalize": "finalize",
            "regenerate_strict": "regenerate_strict",
            "abstain": "abstain",
        },
    )
    builder.add_edge("answer_general", "finalize")
    builder.add_edge("clarify", "finalize")
    builder.add_edge("refuse", "finalize")
    builder.add_edge("abstain", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


def _initial_state(question: str, history: list[dict[str, str]] | None) -> dict[str, Any]:
    return {
        "question": question,
        "history": list(history or []),
        "rewrites": 0,
        "regen_count": 0,
        "filters": {},
        "timings": [],
        "trace": [],
    }


def iter_graph(
    rt: AgentRuntime, question: str, *, history: list[dict[str, str]] | None = None
) -> Iterator[GraphTick]:
    """Yield `start`/`end` per node, then `done` with the final state.

    Uses LangGraph `stream_mode=["tasks", "values"]` so the UI can show the
    node that is running, not only the one that just finished.
    """
    graph = build_graph(Nodes(rt))
    last: RAGState | None = None
    for mode, data in graph.stream(
        _initial_state(question, history),
        {"recursion_limit": 40},
        stream_mode=["tasks", "values"],
    ):
        if mode == "values":
            last = data
            continue
        if mode != "tasks" or not isinstance(data, dict):
            continue
        name = str(data.get("name") or "")
        if not name or name in _SKIP_NODES:
            continue
        # start payload has `input`; result payload has `result` or `error`
        kind: Literal["start", "end"] = "end" if ("result" in data or "error" in data) else "start"
        yield GraphTick(kind=kind, node=name)
    yield GraphTick(kind="done", state=last)


def run_graph(
    rt: AgentRuntime, question: str, *, history: list[dict[str, str]] | None = None
) -> RAGState:
    last: RAGState | None = None
    for tick in iter_graph(rt, question, history=history):
        if tick.kind == "done":
            last = tick.state
    if last is None:
        raise RuntimeError("graph finished without a values snapshot")
    return last
