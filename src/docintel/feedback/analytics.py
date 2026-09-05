from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy import select

from docintel.feedback.models import DocumentRow, FeedbackRow, QueryLogRow
from docintel.feedback.repository import SqlAlchemyFeedbackRepository


def ratings_by_route(repo: SqlAlchemyFeedbackRepository) -> list[dict[str, Any]]:
    return _group(repo, key=lambda _fb, log: log.route or "(none)")


def ratings_by_config_hash(repo: SqlAlchemyFeedbackRepository) -> list[dict[str, Any]]:
    return _group(repo, key=lambda _fb, log: log.config_hash)


def ratings_by_agreement_type(repo: SqlAlchemyFeedbackRepository) -> list[dict[str, Any]]:
    types = _doc_types(repo)
    buckets: dict[str, list[int]] = defaultdict(list)
    with repo.session() as session:
        stmt = select(FeedbackRow, QueryLogRow).join(
            QueryLogRow, FeedbackRow.query_id == QueryLogRow.query_id
        )
        for fb, log in session.execute(stmt).all():
            names = [types.get(doc_id, "(unknown)") for doc_id in (log.cited_doc_ids or [])]
            if not names:
                names = ["(none)"]
            for name in dict.fromkeys(names):
                buckets[name].append(fb.rating)
    return [_summary(name, vals) for name, vals in sorted(buckets.items())]


def worst_queries(repo: SqlAlchemyFeedbackRepository, *, n: int = 20) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    with repo.session() as session:
        stmt = select(FeedbackRow, QueryLogRow).join(
            QueryLogRow, FeedbackRow.query_id == QueryLogRow.query_id
        )
        for fb, log in session.execute(stmt).all():
            slot = grouped.setdefault(
                log.query_id,
                {
                    "query_id": log.query_id,
                    "question": log.question,
                    "route": log.route,
                    "profile": log.profile,
                    "ratings": [],
                },
            )
            slot["ratings"].append(fb.rating)
    out: list[dict[str, Any]] = []
    for slot in grouped.values():
        ratings = slot.pop("ratings")
        out.append(
            {
                **slot,
                "n": len(ratings),
                "mean_rating": sum(ratings) / len(ratings),
            }
        )
    out.sort(key=lambda row: (row["mean_rating"], -row["n"]))
    return out[:n]


def export_csv(repo: SqlAlchemyFeedbackRepository, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    types = _doc_types(repo)
    with repo.session() as session, path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "feedback_id",
                "query_id",
                "created_at",
                "rating",
                "tags",
                "comment",
                "question",
                "route",
                "profile",
                "config_hash",
                "agreement_types",
                "abstained",
            ],
        )
        writer.writeheader()
        stmt = select(FeedbackRow, QueryLogRow).join(
            QueryLogRow, FeedbackRow.query_id == QueryLogRow.query_id
        )
        for fb, log in session.execute(stmt).all():
            names = [types.get(doc_id, "(unknown)") for doc_id in (log.cited_doc_ids or [])]
            writer.writerow(
                {
                    "feedback_id": fb.feedback_id,
                    "query_id": fb.query_id,
                    "created_at": fb.created_at,
                    "rating": fb.rating,
                    "tags": ",".join(fb.tags or []),
                    "comment": fb.comment or "",
                    "question": log.question,
                    "route": log.route or "",
                    "profile": log.profile,
                    "config_hash": log.config_hash,
                    "agreement_types": ",".join(dict.fromkeys(names)),
                    "abstained": log.abstained,
                }
            )
    return path


def _doc_types(repo: SqlAlchemyFeedbackRepository) -> dict[str, str]:
    with repo.session() as session:
        return {row.doc_id: row.agreement_type for row in session.scalars(select(DocumentRow))}


def _group(
    repo: SqlAlchemyFeedbackRepository,
    *,
    key: Any,
) -> list[dict[str, Any]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    with repo.session() as session:
        stmt = select(FeedbackRow, QueryLogRow).join(
            QueryLogRow, FeedbackRow.query_id == QueryLogRow.query_id
        )
        for fb, log in session.execute(stmt).all():
            buckets[str(key(fb, log))].append(fb.rating)
    return [_summary(name, vals) for name, vals in sorted(buckets.items())]


def _summary(name: str, ratings: list[int]) -> dict[str, Any]:
    return {
        "key": name,
        "n": len(ratings),
        "mean_rating": (sum(ratings) / len(ratings)) if ratings else 0.0,
        "up": sum(1 for r in ratings if r > 0),
        "down": sum(1 for r in ratings if r < 0),
    }
