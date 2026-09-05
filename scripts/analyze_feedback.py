from __future__ import annotations

import argparse
from pathlib import Path

from docintel.config import load_config
from docintel.config.loader import find_repo_root
from docintel.feedback.analytics import (
    export_csv,
    ratings_by_agreement_type,
    ratings_by_config_hash,
    ratings_by_route,
    worst_queries,
)
from docintel.feedback.repository import SqlAlchemyFeedbackRepository

ROOT = Path(__file__).resolve().parents[1]


def _table(title: str, rows: list[dict[str, object]], keys: list[str]) -> str:
    lines = [title, " | ".join(keys)]
    for row in rows:
        lines.append(" | ".join(str(row.get(k, "")) for k in keys))
    if not rows:
        lines.append("(none)")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Feedback analytics tables / CSV")
    parser.add_argument("--profile", default="gpu_default")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--worst", type=int, default=20)
    args = parser.parse_args()
    root = find_repo_root(ROOT)
    cfg = load_config(args.profile, repo_root=root)
    # CWD-relative on purpose: must open the same docintel.db the ingest registry wrote
    repo = SqlAlchemyFeedbackRepository(cfg.feedback.db_url)
    try:
        cols = ["key", "n", "mean_rating", "up", "down"]
        print(_table("by route", ratings_by_route(repo), cols))
        print()
        print(_table("by agreement type", ratings_by_agreement_type(repo), cols))
        print()
        print(_table("by config hash", ratings_by_config_hash(repo), cols))
        print()
        print(
            _table(
                "worst queries",
                worst_queries(repo, n=args.worst),
                ["query_id", "mean_rating", "n", "route", "question"],
            )
        )
        if args.csv is not None:
            out = args.csv if args.csv.is_absolute() else root / args.csv
            print()
            print(f"csv={export_csv(repo, out)}")
    finally:
        repo.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
