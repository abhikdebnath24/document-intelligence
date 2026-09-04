from __future__ import annotations

from pathlib import Path

import pytest

from docintel.config import load_config
from docintel.evaluation.experiment import FinalistGateError, load_finalists, run_retrieval_eval

ROOT = Path(__file__).resolve().parents[2]


def test_load_finalists_skips_comments(tmp_path: Path) -> None:
    path = tmp_path / "finalists.txt"
    path.write_text("# locked\nexp_hybrid_rrf\n\n", encoding="utf-8")
    assert load_finalists(path) == {"exp_hybrid_rrf"}


def test_test_split_refuses_unlocked_profile() -> None:
    cfg = load_config("exp_dense_only", repo_root=ROOT)
    with pytest.raises(FinalistGateError, match="not in"):
        run_retrieval_eval(cfg, split="test", repo_root=ROOT)


def test_unknown_split_rejected() -> None:
    cfg = load_config("exp_dense_only", repo_root=ROOT)
    with pytest.raises(ValueError, match="dev or test"):
        run_retrieval_eval(cfg, split="foo", repo_root=ROOT)
