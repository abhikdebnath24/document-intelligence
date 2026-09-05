from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from docintel.agent.nodes import Nodes
from docintel.agent.runtime import AgentRuntime
from docintel.agent.state import RAGState


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


def run_graph(
    rt: AgentRuntime, question: str, *, history: list[dict[str, str]] | None = None
) -> RAGState:
    nodes = Nodes(rt)
    graph = build_graph(nodes)
    result = graph.invoke(
        {
            "question": question,
            "history": list(history or []),
            "rewrites": 0,
            "regen_count": 0,
            "filters": {},
            "timings": [],
            "trace": [],
        },
        {"recursion_limit": 40},
    )
    return result  # type: ignore[no-any-return]
