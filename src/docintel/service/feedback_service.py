from __future__ import annotations

import uuid
from typing import Any

from docintel.core.interfaces import BaseFeedbackRepository
from docintel.core.types import Feedback, QueryLog
from docintel.feedback.models import ALLOWED_RATINGS, FEEDBACK_TAGS


class FeedbackService:
    def __init__(self, repo: BaseFeedbackRepository) -> None:
        self.repo = repo

    def log_query(self, log: QueryLog) -> str:
        return self.repo.log_query(log)

    def rate(
        self,
        query_id: str,
        rating: int,
        *,
        tags: list[str] | None = None,
        comment: str | None = None,
        corrected_citation: dict[str, Any] | None = None,
        feedback_id: str | None = None,
    ) -> Feedback:
        if rating not in ALLOWED_RATINGS:
            raise ValueError(f"rating must be one of {sorted(ALLOWED_RATINGS)}, got {rating!r}")
        clean = list(tags or [])
        unknown = [t for t in clean if t not in FEEDBACK_TAGS]
        if unknown:
            raise ValueError(f"unknown feedback tags: {unknown}; allowed: {sorted(FEEDBACK_TAGS)}")
        item = Feedback(
            feedback_id=feedback_id or str(uuid.uuid4()),
            query_id=query_id,
            rating=rating,
            tags=clean,
            comment=comment,
            corrected_citation=corrected_citation,
        )
        self.repo.add_feedback(item)
        return item
