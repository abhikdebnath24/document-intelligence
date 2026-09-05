from __future__ import annotations

import argparse
from pathlib import Path

from docintel.config import load_config
from docintel.config.loader import find_repo_root
from docintel.evaluation.experiment import FinalistGateError
from docintel.evaluation.generation_eval import run_generation_eval

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="L2 generation eval (RAGAS / DeepEval / custom)")
    parser.add_argument("--profile", default="gpu_default")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument(
        "--framework",
        default="all",
        choices=["all", "ragas", "deepeval", "custom"],
        help="custom skips LLM judges; all runs ragas+deepeval+custom",
    )
    args = parser.parse_args()
    root = find_repo_root(ROOT)
    cfg = load_config(args.profile, repo_root=root)
    frameworks = {
        "all": ["ragas", "deepeval"],
        "ragas": ["ragas"],
        "deepeval": ["deepeval"],
        "custom": [],
    }[args.framework]
    try:
        path = run_generation_eval(cfg, split=args.split, repo_root=root, frameworks=frameworks)
    except FinalistGateError as exc:
        raise SystemExit(str(exc)) from exc
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
