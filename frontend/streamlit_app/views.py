from __future__ import annotations

import json
import uuid

import streamlit as st
from client import InProcessClient, RagClient, run_pdf_ingest

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
from docintel.service.ingest_service import UploadError
from docintel.service.pdf_render import highlight_boxes, render_cited_page


def chat_page(client: RagClient) -> None:
    st.title("Document Intelligence")
    st.caption("Ask the Corpus")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    question = st.chat_input("Ask a contract question, or anything the model knows")
    for msg in st.session_state.messages:
        _replay(client, msg)
    if question:
        _run_turn(client, question)


def _run_turn(client: RagClient, question: str) -> None:
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.write(question)

    history = _history(st.session_state.messages[:-1])
    steps: list[str] = []
    answer: Answer | None = None
    log: QueryLog | None = None
    box = st.status("Working...", expanded=False)
    try:
        for tick in client.ask_iter(
            question, session_id=st.session_state.session_id, history=history
        ):
            if tick.kind == "start":
                steps.append(tick.node)
                box.update(label=step_label(tick.node), state="running")
                box.write(step_label(tick.node))
            elif tick.kind == "done" and tick.state is not None:
                box.update(label=f"Done · {len(steps)} steps", state="complete")
                answer = tick.state["answer"]
                log = tick.state["log"]
    except Exception as exc:
        box.update(label="Graph failed", state="error")
        st.error(str(exc))
        return
    if answer is None or log is None:
        box.update(label="Graph returned no answer", state="error")
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
    names = [str(node) for node in steps] if isinstance(steps, list) else []
    if names:
        with st.status(f"Done · {len(names)} steps", state="complete", expanded=False):
            for node in names:
                st.write(step_label(node))
    _render_answer(client, msg)


def _render_answer(client: RagClient, msg: dict[str, object]) -> None:
    answer: Answer = msg["answer"]  # type: ignore[assignment]
    log: QueryLog = msg["log"]  # type: ignore[assignment]
    with st.chat_message("assistant"):
        bits = [answer.route]
        if answer.abstained:
            bits.append("abstained")
        meta = " · ".join(bits)
        if answer.groundedness is not None:
            meta += f" · groundedness {answer.groundedness:.2f}"
        meta += f" · {log.latency_ms} ms"
        st.caption(meta)
        st.write(answer.text)
        if answer.citations:
            st.markdown("**Citations**")
            for i, cite in enumerate(answer.citations):
                label = f"{cite.doc_id or cite.chunk_id}  p.{cite.page_no}"
                if st.button(label, key=f"cite-{log.query_id}-{i}", width="stretch"):
                    st.session_state.selected_cite = cite
                    _cite_dialog(client)
        _feedback_widget(log.query_id)
        with st.expander("Trace JSON"):
            st.json(
                {
                    "query_id": log.query_id,
                    "rewrites": log.rewrites,
                    "llm_calls": log.llm_calls,
                    "trace_path": log.trace_path,
                    "timings": [t.model_dump() for t in answer.timings],
                }
            )


def _feedback_widget(query_id: str) -> None:
    client: RagClient = st.session_state.client
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


@st.dialog("Source", width="large")
def _cite_dialog(client: RagClient) -> None:
    cite = st.session_state.get("selected_cite")
    if not isinstance(cite, Citation):
        st.caption("Click a citation to open the PDF page.")
        return
    if not cite.doc_id:
        st.info(cite.quote or "No document id on this citation.")
        return
    path = client.find_source(cite.doc_id)
    if path is None:
        st.warning("PDF not on disk. Quote fallback:")
        st.write(cite.quote)
        return
    png = render_cited_page(path, cite, dpi=client.config.frontend.pdf_render_dpi)
    note = ""
    if not highlight_boxes(cite):
        note = (
            " · multi-page chunk, highlight off"
            if (cite.page_end or cite.page_no) != cite.page_no
            else " · no bboxes"
        )
    st.image(png, caption=f"{path.name}  p.{cite.page_no}{note}", width="stretch")
    if cite.quote:
        st.caption(cite.quote)


def documents_page(client: RagClient) -> None:
    st.title("Documents")
    st.caption("Drop a contract PDF. Incremental ingest upserts it into the active collection.")
    _upload_block(client)
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


def _upload_block(client: RagClient) -> None:
    uploaded = st.file_uploader("Upload PDF", type=["pdf"], key="docs_pdf_upload")
    result = st.session_state.get("last_ingest")
    if isinstance(result, dict):
        c1, c2, c3 = st.columns(3)
        c1.metric("Chunks", result.get("chunks"))
        c2.metric("Indexed", result.get("indexed"))
        c3.metric("Seconds", result.get("seconds"))
        st.caption(
            f"doc_id={result.get('doc_id')}  collection={result.get('collection')}  "
            f"skipped={result.get('skipped')}"
        )
    if error := st.session_state.pop("ingest_error", None):
        st.error(error)
        if st.button("Retry upload"):
            st.session_state.pop("_ingest_token", None)
            st.rerun()
    if uploaded is None:
        return
    token = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("_ingest_token") == token:
        return
    with st.spinner("Loading, chunking, embedding, upserting...", show_time=True):
        try:
            st.session_state.last_ingest = run_pdf_ingest(
                client, uploaded.getvalue(), filename=uploaded.name
            )
            st.session_state._ingest_token = token
        except UploadError as exc:
            st.session_state.ingest_error = str(exc)
            st.session_state._ingest_token = token
        except Exception as exc:
            st.session_state.ingest_error = f"Ingest failed: {exc}"
            st.session_state._ingest_token = token
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
