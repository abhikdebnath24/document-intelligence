from __future__ import annotations

import argparse
from pathlib import Path

from docintel.config import load_config
from docintel.config.loader import find_repo_root
from docintel.service.container import Container
from docintel.service.query_service import QueryService

ROOT = Path(__file__).resolve().parents[1]

# 6 representative paths. No cross_ref bucket in qa_dev; multi-span slot stands in.
DEMO = [
    ("general", "What does a governing-law clause usually decide?"),
    (
        "slot",
        "Which jurisdiction's law governs the Telkomsaltd LICENCE AND MAINTENANCE AGREEMENT?",
    ),
    ("yes_span", "Does the Cheetahmobileinc Cooperation Agreement include an Exclusivity clause?"),
    ("abstain", "Does the Nelnetinc JOINT FILING AGREEMENT include an Exclusivity clause?"),
    ("ambiguous", "Does this agreement include an Exclusivity clause?"),
    (
        "slot",
        "Which jurisdiction's law governs the Meetgroup,Inc COOPERATION AGREEMENT?",
    ),
]


def render_demo(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Demo queries",
        "",
        "Faithfulness cells stay empty until `--annotate` on a live L2 run.",
        "",
    ]
    for row in rows:
        lines += [
            f"## {row['bucket']}",
            "",
            f"**Q.** {row['question']}",
            "",
            f"**route** `{row['route']}` **abstained** `{row['abstained']}` "
            f"**groundedness** `{row['groundedness']}`",
            "",
            row["answer"] or "_(empty)_",
            "",
            "**contexts**",
            "",
        ]
        contexts = row.get("contexts") or []
        if not contexts:
            lines += ["_(none)_", ""]
        for i, ctx in enumerate(contexts[:3], start=1):
            snippet = str(ctx).replace("\n", " ")[:400]
            lines += [f"{i}. {snippet}", ""]
        lines += ["**faithfulness** _pending_", "", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Write results/demo_queries.md")
    parser.add_argument("--profile", default="gpu_default")
    parser.add_argument("--out", default="results/demo_queries.md")
    args = parser.parse_args()
    root = find_repo_root(ROOT)
    cfg = load_config(args.profile, repo_root=root)
    svc = QueryService(Container(cfg, repo_root=root))
    rows: list[dict[str, object]] = []
    try:
        for bucket, question in DEMO:
            answer, log = svc.ask(question)
            rows.append(
                {
                    "bucket": bucket,
                    "question": question,
                    "route": answer.route,
                    "abstained": answer.abstained,
                    "groundedness": answer.groundedness,
                    "answer": answer.text,
                    "contexts": log.retrieved_contexts,
                }
            )
    finally:
        svc.close()
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_demo(rows), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
