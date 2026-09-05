from __future__ import annotations

from pathlib import Path

from frontend.streamlit_app.client import InProcessClient

from docintel.config import load_config
from docintel.core.types import QueryLog
from docintel.feedback.repository import SqlAlchemyFeedbackRepository
from docintel.service.feedback_service import FeedbackService

ROOT = Path(__file__).resolve().parents[2]


def test_client_is_lazy_and_updates_one_feedback_row() -> None:
    config = load_config("dev_cpu", repo_root=ROOT)
    repo = SqlAlchemyFeedbackRepository("sqlite:///:memory:")
    feedback = FeedbackService(repo)
    client = InProcessClient(config, feedback, ROOT)
    assert client._query is None

    feedback.log_query(
        QueryLog(
            query_id="q1",
            question="Which law governs?",
            route="corpus_technical",
            config_hash="abc",
            profile="dev_cpu",
            answer="Delaware",
        )
    )
    client.rate("q1", 3, tags=["incomplete"])
    client.rate("q1", 5, tags=["good"], comment="fixed")

    rows = repo.list_feedback("q1")
    assert len(rows) == 1
    assert rows[0].rating == 5
    assert rows[0].tags == ["good"]
    assert rows[0].comment == "fixed"
    client.close()
