from __future__ import annotations

import json
import uuid

import streamlit as st
from client import InProcessClient, RagClient

from docintel.agent.graph import step_label
from docintel.core.types import Answer, Citation, QueryLog
from docintel.feedback.analytics import (
    ratings_by_agreement_type,
    ratings_by_config_hash,
    ratings_by_route,
    worst_queries,
)
from docintel.feedback.models import FEEDBACK_TAGS
from docintel.feedback.repository import SqlAlchemyFeedbackRepository
from docintel.service.ingest_service import UploadError, save_upload
from docintel.service.pdf_render import highlight_boxes, render_cited_page

NOTICE = (
    "With an Anthropic / OpenAI / Google key set, questions, answers, and retrieved "
    "clause text leave this machine."
)


def chat_page(client: RagClient) -> None:
    st.title("Ask the corpus")
    st.caption(NOTICE)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    for msg in st.session_state.messages:
        _replay(client, msg)

    question = st.chat_input("Ask about a contract clause")
    if not question:
        return
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.write(question)

    history = _history(st.session_state.messages[:-1])
    steps: list[str] = []
    open_step = None
    answer: Answer | None = None
    log: QueryLog | None = None
    try:
        for tick in client.ask_iter(
            question, session_id=st.session_state.session_id, history=history
        ):
            if tick.kind == "start":
                if open_step is not None:
                    open_step.update(state="complete")
                open_step = st.status(
                    step_label(tick.node),
                    state="running",
                    type="step",
                    expanded=True,
                )
                open_step.write(f"`{tick.node}`")
                steps.append(tick.node)
            elif tick.kind == "end" and open_step is not None:
                open_step.update(state="complete")
            elif tick.kind == "done" and tick.state is not None:
                if open_step is not None:
                    open_step.update(state="complete")
                st.status("Answer ready", state="complete", type="step")
                answer = tick.state["answer"]
                log = tick.state["log"]
    except Exception as exc:
        if open_step is not None:
            open_step.update(label="Graph failed", state="error")
        st.error(str(exc))
        return
    if answer is None or log is None:
        st.error("Graph returned no answer.")
        return
    record = {"role": "assistant", "answer": answer, "log": log, "steps": steps}
    st.session_state.messages.append(record)
    _render_answer(client, record)


def _replay(client: RagClient, msg: dict[str, object]) -> None:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(str(msg["text"]))
        return
    steps = msg.get("steps")
    for node in steps if isinstance(steps, list) else []:
        with st.status(step_label(str(node)), state="complete", type="step"):
            st.write(f"`{node}`")
    if steps:
        st.status("Answer ready", state="complete", type="step")
    _render_answer(client, msg)


def _render_answer(client: RagClient, msg: dict[str, object]) -> None:
    answer: Answer = msg["answer"]  # type: ignore[assignment]
    log: QueryLog = msg["log"]  # type: ignore[assignment]
    with st.chat_message("assistant"):
        st.badge(answer.route, color="blue")
        if answer.abstained:
            st.badge("abstained", color="orange")
        if answer.groundedness is not None:
            st.caption(f"groundedness {answer.groundedness:.2f} · {log.latency_ms} ms")
        st.write(answer.text)
        if answer.citations:
            st.subheader("Citations")
            for i, cite in enumerate(answer.citations):
                label = f"{cite.doc_id or cite.chunk_id} p.{cite.page_no}"
                if st.button(label, key=f"cite-{log.query_id}-{i}"):
                    st.session_state.selected_cite = cite
                    st.session_state.selected_query_id = log.query_id
                st.caption(cite.quote)
        _feedback_widget(client, log.query_id)
        with st.expander("Trace"):
            st.json(
                {
                    "query_id": log.query_id,
                    "rewrites": log.rewrites,
                    "llm_calls": log.llm_calls,
                    "trace_path": log.trace_path,
                    "timings": [t.model_dump() for t in answer.timings],
                }
            )
        selected = st.session_state.get("selected_cite")
        if (
            isinstance(selected, Citation)
            and st.session_state.get("selected_query_id") == log.query_id
        ):
            _pdf_panel(client, selected)


def _feedback_widget(client: RagClient, query_id: str) -> None:
    key = f"fb-{query_id}"
    with st.form(key, border=False):
        stars = st.feedback("stars", key=f"{key}-stars")
        tags = st.multiselect("tags", sorted(FEEDBACK_TAGS), key=f"{key}-tags")
        comment = st.text_input("comment", key=f"{key}-comment")
        submitted = st.form_submit_button("Save feedback")
    if submitted:
        if stars is None:
            st.warning("Select a star rating first.")
            return
        rating = stars + 1
        client.rate(query_id, rating, tags=tags, comment=comment or None)
        st.session_state[f"{key}-saved"] = rating
        st.success(f"Saved rating {rating}")
    elif saved := st.session_state.get(f"{key}-saved"):
        st.caption(f"Saved rating {saved}")


def _pdf_panel(client: RagClient, cite: Citation) -> None:
    st.subheader("Source page")
    if not cite.doc_id:
        st.info(cite.quote or "No document id on this citation.")
        return
    path = client.find_source(cite.doc_id)
    if path is None:
        st.warning("PDF not on disk. Quote fallback:")
        st.write(cite.quote)
        return
    boxes = highlight_boxes(cite)
    if not boxes:
        reason = "multi-page chunk" if (cite.page_end or cite.page_no) != cite.page_no else "no bboxes"
        st.caption(f"{path.name} p.{cite.page_no} ({reason}; quote only)")
        st.write(cite.quote)
        return
    png = render_cited_page(path, cite, dpi=client.config.frontend.pdf_render_dpi)
    st.image(png, caption=f"{path.name} p.{cite.page_no}")


def documents_page(client: RagClient) -> None:
    st.title("Documents")
    st.caption(NOTICE)
    if notice := st.session_state.pop("ingest_notice", None):
        st.success(notice)
    rows = client.list_documents()
    if rows:
        st.dataframe(
            [
                {
                    "doc_id": r.doc_id,
                    "type": r.agreement_type,
                    "chunks": r.n_chunks,
                    "status": r.status,
                    "split": r.split,
                    "path": r.source_path,
                }
                for r in rows
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No indexed documents in docintel.db yet.")

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded is None:
        return
    if st.button("Ingest upload"):
        dest = client.repo_root / "data" / "uploads"
        try:
            path = save_upload(uploaded.getvalue(), dest)
            with st.spinner("Loading, chunking, embedding, upserting...", show_time=True):
                report = client.ingest_paths([path])
        except UploadError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Ingest failed: {exc}")
            return
        if report.failed:
            st.error(report.failed)
            return
        st.session_state.ingest_notice = (
            f"indexed={report.indexed} skipped={report.skipped} "
            f"chunks={report.chunks} collection={report.collection}"
        )
        st.rerun()


def experiments_page(client: RagClient) -> None:
    st.title("Experiments")
    root = client.repo_root / "results"
    st.caption("L1 / L2 folders under results/. MLflow UI is a separate process.")
    st.link_button("Open MLflow UI", "http://127.0.0.1:5000")
    readme = root / "README.md"
    if readme.is_file():
        st.markdown(readme.read_text(encoding="utf-8"))
    runs = sorted(p for p in root.glob("*") if p.is_dir())
    if not runs:
        st.info("No results/ folders yet.")
        return
    names = [p.name for p in runs]
    picked = st.multiselect("Compare runs", names, default=names[: min(3, len(names))])
    table: list[dict[str, object]] = []
    for name in picked:
        folder = root / name
        l1 = folder / "retrieval_metrics.json"
        l2 = folder / "generation_metrics.json"
        row: dict[str, object] = {"run": name}
        if l1.is_file():
            payload = json.loads(l1.read_text())
            overall = payload.get("overall") or {}
            row.update(
                {
                    "hit@5": overall.get("hit@5"),
                    "r@10": overall.get("r@10"),
                    "ndcg@10": overall.get("ndcg@10"),
                    "mrr": overall.get("mrr"),
                }
            )
        if l2.is_file():
            payload = json.loads(l2.read_text())
            ragas = payload.get("ragas") or {}
            custom = payload.get("custom") or {}
            row.update(
                {
                    "faithfulness": ragas.get("faithfulness"),
                    "route_accuracy": custom.get("route_accuracy"),
                    "latency_p50_ms": custom.get("latency_p50_ms"),
                }
            )
        table.append(row)
    if table:
        st.dataframe(table, hide_index=True, width="stretch")


def analytics_page(client: RagClient) -> None:
    st.title("Feedback analytics")
    if not isinstance(client, InProcessClient):
        st.info("Analytics reads the local SQLite feedback store.")
        return
    repo = client.feedback.repo
    if not isinstance(repo, SqlAlchemyFeedbackRepository):
        st.info("Analytics requires the local SQLite feedback store.")
        return
    for title, rows in (
        ("By route", ratings_by_route(repo)),
        ("By config hash", ratings_by_config_hash(repo)),
        ("By agreement type", ratings_by_agreement_type(repo)),
    ):
        st.subheader(title)
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
        else:
            st.caption("No ratings yet.")
    st.subheader("Worst queries")
    worst = worst_queries(repo)
    if worst:
        st.dataframe(worst, hide_index=True, width="stretch")
    else:
        st.caption("No rated queries yet.")


def _history(messages: list[dict[str, object]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for msg in messages[-6:]:
        if msg["role"] == "user":
            out.append({"role": "user", "content": str(msg["text"])})
        else:
            answer: Answer = msg["answer"]  # type: ignore[assignment]
            out.append({"role": "assistant", "content": answer.text})
    return out
