from __future__ import annotations

import argparse
from pathlib import Path

from docintel.config import load_config
from docintel.config.loader import find_repo_root
from docintel.evaluation.experiment import FinalistGateError, run_retrieval_eval

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="L1 retrieval eval")
    parser.add_argument("--profile", default="exp_hybrid_rrf")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument(
        "--scoped",
        action="store_true",
        help="diagnostic: force gold doc_id filter (not the scored path)",
    )
    args = parser.parse_args()
    root = find_repo_root(ROOT)
    cfg = load_config(args.profile, repo_root=root)
    try:
        path = run_retrieval_eval(cfg, split=args.split, repo_root=root, scoped=args.scoped)
    except FinalistGateError as exc:
        raise SystemExit(str(exc)) from exc
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
