from __future__ import annotations

from typing import cast

import streamlit as st
from client import HttpClient, InProcessClient, RagClient
from views import (
    analytics_page,
    chat_page,
    documents_page,
    experiments_page,
)

from docintel.config import load_config
from docintel.config.loader import find_repo_root
from docintel.feedback.repository import SqlAlchemyFeedbackRepository
from docintel.service.feedback_service import FeedbackService
from docintel.settings import load_settings

st.set_page_config(page_title="docintel", layout="wide")


def _profiles() -> list[str]:
    folder = find_repo_root() / "configs" / "profiles"
    return sorted(p.stem for p in folder.glob("*.yaml"))


# Process-global on purpose. Embedded Qdrant is one QdrantClient per path shared by
# every QdrantIndexer in this process (`_CLIENTS`); a session-scoped cache would let
# tab A's on_release close the client tab B's LazyPipeline still holds.
# max_entries=1: one live client per process; a profile switch evicts and closes it.
@st.cache_resource(max_entries=1, on_release=lambda client: client.close())
def boot(profile: str) -> RagClient:
    cfg = load_config(profile)
    if cfg.frontend.backend == "http":
        return cast(RagClient, HttpClient(cfg.frontend.api_url))
    root = find_repo_root()
    feedback = FeedbackService(SqlAlchemyFeedbackRepository(cfg.feedback.db_url))
    return InProcessClient(cfg, feedback, root)


def page_chat() -> None:
    chat_page(st.session_state.client)


def page_documents() -> None:
    documents_page(st.session_state.client)


def page_experiments() -> None:
    experiments_page(st.session_state.client)


def page_feedback() -> None:
    analytics_page(st.session_state.client)


profiles = _profiles()
default = load_settings().docintel_profile
idx = profiles.index(default) if default in profiles else 0
profile = st.sidebar.selectbox("profile", profiles, index=idx)

if st.session_state.get("active_profile") != profile:
    for key in ("messages", "selected_cite", "selected_query_id"):
        st.session_state.pop(key, None)
    st.session_state.active_profile = profile

try:
    st.session_state.client = boot(profile)
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.sidebar.caption(f"backend={st.session_state.client.config.frontend.backend}")

page = st.navigation(
    {
        "Query": [st.Page(page_chat, title="Chat", default=True)],
        "Corpus": [st.Page(page_documents, title="Documents")],
        "Eval": [
            st.Page(page_experiments, title="Experiments"),
            st.Page(page_feedback, title="Feedback"),
        ],
    }
)
page.run()
