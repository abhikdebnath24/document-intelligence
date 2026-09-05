from __future__ import annotations

import time
from typing import Any

from docintel.agent.edges import after_classify, after_grade, after_verify
from docintel.agent.runtime import AgentRuntime
from docintel.agent.state import RAGState
from docintel.agent.verifiers import validate_citations
from docintel.core.types import Answer, Citation, RetrievalQuery, RetrievedChunk
from docintel.llm.schemas import ClarifyOut, GenerateOut, RewriteOut, RouteOut


class Nodes:
    def __init__(self, rt: AgentRuntime) -> None:
        self.rt = rt

    def classify_query(self, state: RAGState) -> dict[str, Any]:
        t0 = time.monotonic()
        self.rt.check_deadline()
        out = self.rt.caller.structured(
            "router",
            RouteOut,
            f"Question: {state['question']}",
            system=self.rt.prompts.get("classify").text,
        )
        # LLM hints are for messages only. Retrieval filters stay with the deterministic
        # FilterExtractor inside RetrievalPipeline.search; an unknown payload key or a
        # mis-cased type from the router would return zero points and force an abstain.
        return _delta(
            "classify_query",
            t0,
            {
                "route": out.route,
                "route_reason": out.reason,
                "doc_hint": out.doc_hint or "",
            },
            prompt_version=self.rt.prompts.get("classify").version,
        )

    def plan_retrieval(self, state: RAGState) -> dict[str, Any]:
        t0 = time.monotonic()
        queries = list(state.get("search_queries") or []) or [state["question"]]
        return _delta("plan_retrieval", t0, {"search_queries": queries})

    def retrieve_hybrid(self, state: RAGState) -> dict[str, Any]:
        t0 = time.monotonic()
        self.rt.check_deadline()
        k = self.rt.config.retrieval.k_candidates
        filters = dict(state.get("filters") or {})
        rows: list[RetrievedChunk] = []
        for text in state.get("search_queries") or [state["question"]]:
            rows.extend(self.rt.pipeline.search(RetrievalQuery(text=text, k=k, filters=filters)))
        # keep earlier candidates: a rewrite must not drop chunks the first pass found
        rows.extend(self.rt.cache.get_many(list(state.get("candidate_ids") or [])))
        by_id: dict[str, RetrievedChunk] = {}
        for row in rows:
            by_id.setdefault(row.chunk.chunk_id, row)
        ids = self.rt.cache.put(list(by_id.values()))
        return _delta("retrieve_hybrid", t0, {"candidate_ids": ids}, n=len(ids))

    def rerank(self, state: RAGState) -> dict[str, Any]:
        t0 = time.monotonic()
        rows = self.rt.cache.get_many(list(state.get("candidate_ids") or []))
        ranked = self.rt.pipeline.reranker.rerank(state["question"], rows, len(rows) or 1)
        ids = self.rt.cache.put(ranked)
        return _delta("rerank", t0, {"ranked_ids": ids}, n=len(ids))

    def grade_documents(self, state: RAGState) -> dict[str, Any]:
        t0 = time.monotonic()
        self.rt.check_deadline()
        rows = self.rt.cache.get_many(list(state.get("ranked_ids") or []))
        grades = self.rt.grader.grade(state["question"], rows)
        relevant = [g.chunk_id for g in grades if g.relevant]
        return _delta(
            "grade_documents",
            t0,
            {
                "grades": [g.model_dump() for g in grades],
                "relevant_ids": relevant,
            },
            n_relevant=len(relevant),
        )

    def rewrite_query(self, state: RAGState) -> dict[str, Any]:
        t0 = time.monotonic()
        self.rt.check_deadline()
        reasons = "; ".join(
            f"{g.get('chunk_id')}: {g.get('reason')}" for g in (state.get("grades") or [])[:8]
        )
        prompt = f"Question: {state['question']}\nGrader notes: {reasons}"
        out = self.rt.caller.structured(
            "router", RewriteOut, prompt, system=self.rt.prompts.get("rewrite").text
        )
        query = out.query.strip() or state["question"]
        return _delta(
            "rewrite_query",
            t0,
            {
                "search_queries": [query],
                "rewrites": int(state.get("rewrites") or 0) + 1,
            },
            prompt_version=self.rt.prompts.get("rewrite").version,
        )

    def generate(self, state: RAGState) -> dict[str, Any]:
        return self._generate(state, strict=False)

    def regenerate_strict(self, state: RAGState) -> dict[str, Any]:
        delta = self._generate(state, strict=True)
        delta["regen_count"] = int(state.get("regen_count") or 0) + 1
        return delta

    def verify_groundedness(self, state: RAGState) -> dict[str, Any]:
        t0 = time.monotonic()
        self.rt.check_deadline()
        draft = GenerateOut.model_validate(
            state.get("draft_answer") or {"answer": "", "citations": []}
        )
        rows = self.rt.cache.get_many(list(state.get("relevant_ids") or []))
        cites = _citations(draft, self.rt.cache)
        cite_errors = validate_citations(
            cites, rows, threshold=self.rt.config.agent.citation_quote_fuzzy_threshold
        )
        result = self.rt.verifier.verify(draft.answer, rows)
        unsupported = list(result.unsupported_claims)
        unsupported.extend(cite_errors)
        grounded = bool(result.grounded) and not cite_errors
        payload: dict[str, Any] = {
            "grounded": grounded,
            "groundedness_score": result.score,
            "unsupported_claims": unsupported,
        }
        if grounded:
            payload["answer"] = _answer(
                self.rt,
                state,
                draft.answer,
                cites,
                abstained=False,
                groundedness=result.score,
            )
        return _delta("verify_groundedness", t0, payload, grounded=grounded)

    def abstain(self, state: RAGState) -> dict[str, Any]:
        t0 = time.monotonic()
        nearest = ""
        if self.rt.config.agent.abstain_show_nearest:
            rows = self.rt.cache.get_many(
                list(state.get("relevant_ids") or state.get("ranked_ids") or [])[:3]
            )
            bits = [f"{row.chunk.doc_id} p.{row.chunk.page_start}" for row in rows]
            if bits:
                nearest = " Closest passages: " + "; ".join(bits) + "."
        doc = state.get("doc_hint") or "the requested document"
        text = f"The knowledge base does not contain a clause answering this for {doc}.{nearest}"
        return _delta(
            "abstain",
            t0,
            {
                "answer": _answer(self.rt, state, text, [], abstained=True, groundedness=0.0),
            },
        )

    def answer_general(self, state: RAGState) -> dict[str, Any]:
        t0 = time.monotonic()
        self.rt.check_deadline()
        body = self.rt.caller.text(
            "generation",
            f"Question: {state['question']}",
            system=self.rt.prompts.get("general").text,
        )
        prefix = "Answered from general knowledge, not from your documents."
        if self.rt.config.agent.general_knowledge_disclaimer and prefix not in body:
            body = f"{prefix} {body}"
        return _delta(
            "answer_general",
            t0,
            {"answer": _answer(self.rt, state, body, [], abstained=False, groundedness=None)},
            prompt_version=self.rt.prompts.get("general").version,
        )

    def clarify(self, state: RAGState) -> dict[str, Any]:
        t0 = time.monotonic()
        self.rt.check_deadline()
        out = self.rt.caller.structured(
            "router",
            ClarifyOut,
            f"Question: {state['question']}",
            system=self.rt.prompts.get("clarify").text,
        )
        return _delta(
            "clarify",
            t0,
            {
                "answer": _answer(
                    self.rt, state, out.question, [], abstained=False, groundedness=None
                )
            },
            prompt_version=self.rt.prompts.get("clarify").version,
        )

    def refuse(self, state: RAGState) -> dict[str, Any]:
        t0 = time.monotonic()
        _ = state
        text = "This question is outside the scope of the contract knowledge base."
        return _delta(
            "refuse",
            t0,
            {"answer": _answer(self.rt, state, text, [], abstained=False, groundedness=None)},
        )

    def finalize(self, state: RAGState) -> dict[str, Any]:
        t0 = time.monotonic()
        answer = dict(state.get("answer") or {})
        answer["trace_id"] = self.rt.query_id
        return _delta("finalize", t0, {"answer": answer})

    def _generate(self, state: RAGState, *, strict: bool) -> dict[str, Any]:
        t0 = time.monotonic()
        self.rt.check_deadline()
        rows = self.rt.cache.get_many(list(state.get("relevant_ids") or []))
        blocks = "\n".join(
            f'<evidence id="{row.chunk.chunk_id}">\n{row.chunk.text}\n</evidence>' for row in rows
        )
        name = "generate_strict" if strict else "generate"
        extra = ""
        if strict and state.get("unsupported_claims"):
            extra = "\nUnsupported claims:\n- " + "\n- ".join(state["unsupported_claims"])
        prompt = f"Question: {state['question']}\n\n{blocks}{extra}"
        out = self.rt.caller.structured(
            "generation", GenerateOut, prompt, system=self.rt.prompts.get(name).text
        )
        return _delta(
            name,
            t0,
            {"draft_answer": out.model_dump()},
            prompt_version=self.rt.prompts.get(name).version,
            n_cites=len(out.citations),
        )

    def route_after_classify(self, state: RAGState) -> str:
        return after_classify(state)

    def route_after_grade(self, state: RAGState) -> str:
        agent = self.rt.config.agent
        return after_grade(
            state, min_relevant=agent.min_relevant_chunks, max_rewrites=agent.max_rewrites
        )

    def route_after_verify(self, state: RAGState) -> str:
        return after_verify(state)


def _citations(draft: GenerateOut, cache: Any) -> list[Citation]:
    out: list[Citation] = []
    for item in draft.citations:
        row = cache.get(item.chunk_id)
        if row is None:
            out.append(
                Citation(chunk_id=item.chunk_id, doc_id="", page_no=0, bboxes=[], quote=item.quote)
            )
            continue
        out.append(
            Citation(
                chunk_id=item.chunk_id,
                doc_id=row.chunk.doc_id,
                page_no=row.chunk.page_start,
                page_end=row.chunk.page_end,
                bboxes=list(row.chunk.bboxes),
                quote=item.quote,
            )
        )
    return out


def _answer(
    rt: AgentRuntime,
    state: RAGState,
    text: str,
    citations: list[Citation],
    *,
    abstained: bool,
    groundedness: float | None,
) -> dict[str, Any]:
    route = state.get("route") or "corpus_technical"
    return Answer(
        text=text,
        citations=citations,
        route=route,
        abstained=abstained,
        groundedness=groundedness,
        trace_id=rt.query_id,
    ).model_dump()


def _delta(node: str, t0: float, payload: dict[str, Any], **summary: object) -> dict[str, Any]:
    elapsed = (time.monotonic() - t0) * 1000
    event = {
        "node": node,
        "t_ms": round(elapsed, 2),
        "outputs_summary": {k: summary[k] for k in summary},
    }
    payload["timings"] = [{"node": node, "t_ms": round(elapsed, 2)}]
    payload["trace"] = [event]
    return payload
