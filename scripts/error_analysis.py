from __future__ import annotations

import argparse
import json
from pathlib import Path

from docintel.evaluation.custom_metrics import top_misses

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Top-20 L1 misses with reason tags")
    parser.add_argument("per_question", type=Path, help="results/<run>/per_question.jsonl")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    path = args.per_question if args.per_question.is_absolute() else ROOT / args.per_question
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    misses = top_misses(rows, n=args.n)
    text = json.dumps(misses, indent=2) + "\n"
    if args.out:
        out = args.out if args.out.is_absolute() else ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(out)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
