from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from docintel.config import AppConfig, config_hash, index_sig
from docintel.config.loader import find_repo_root
from docintel.core.types import RetrievalQuery
from docintel.data.corpus import normalize_stem
from docintel.evaluation.gold import SpanMatcher, file_sha256, load_qa_set
from docintel.evaluation.retrieval_metrics import aggregate, include_item, question_metrics
from docintel.retrieval.factory import build_retrieval_pipeline
from docintel.settings import load_settings


class FinalistGateError(RuntimeError):
    pass


def git_sha(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_finalists(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if text:
            names.add(text)
    return names


def run_retrieval_eval(
    config: AppConfig,
    *,
    split: str,
    repo_root: Path | None = None,
    scoped: bool = False,
) -> Path:
    root = repo_root or find_repo_root()
    load_settings()
    if split not in {"dev", "test"}:
        raise ValueError(f"split must be dev or test, got {split!r}")
    if split == "test":
        allowed = load_finalists(root / config.evaluation.finalists_file)
        if config.profile not in allowed:
            raise FinalistGateError(
                f"--split test refused: {config.profile!r} "
                f"is not in {config.evaluation.finalists_file}"
            )
    qa_rel = config.evaluation.qa_dev if split == "dev" else config.evaluation.qa_test
    qa_path = root / qa_rel
    items = load_qa_set(qa_path)
    matcher = SpanMatcher(config.evaluation.span_match_threshold)
    ks = list(config.evaluation.ks)
    c_hash = config_hash(config)
    i_sig = index_sig(config)
    run_id = f"{config.profile}_{split}_{c_hash[:12]}" + ("_scoped" if scoped else "")
    out_dir = root / config.evaluation.results_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "per_question.jsonl"
    done = _checkpoint_ids(ckpt)
    pipeline = build_retrieval_pipeline(config, repo_root=root)
    try:
        rows: list[dict[str, Any]] = []
        if ckpt.is_file():
            for line in ckpt.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        with ckpt.open("a", encoding="utf-8") as fh:
            for item in items:
                if not include_item(item):
                    continue
                gold = item.model_copy(update={"doc_stem": normalize_stem(item.doc_stem)})
                if gold.id in done:
                    continue
                filters: dict[str, str] = {}
                if scoped and gold.doc_stem:
                    filters["doc_id"] = gold.doc_stem
                hits = pipeline.retrieve(
                    RetrievalQuery(text=gold.question, k=max(ks), filters=filters)
                )
                metrics = question_metrics(gold, hits, matcher, ks)
                rec = {
                    "id": gold.id,
                    "bucket": gold.bucket,
                    "agreement_type": gold.agreement_type,
                    "doc_id": gold.doc_stem,
                    "chunk_ids": [h.chunk.chunk_id for h in hits],
                    "doc_ids": [h.chunk.doc_id for h in hits],
                    **metrics,
                }
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                rows.append(rec)
                done.add(gold.id)
    finally:
        pipeline.close()

    payload = {
        "profile": config.profile,
        "split": split,
        "scoped": scoped,
        "config_hash": c_hash,
        "index_sig": i_sig,
        "qa_sha": file_sha256(qa_path),
        "git_sha": git_sha(root),
        "n": len(rows),
        "overall": aggregate(rows).get("all", {}),
        "by_bucket": aggregate(rows, group_key="bucket"),
        "by_agreement_type": aggregate(rows, group_key="agreement_type"),
    }
    metrics_path = out_dir / "retrieval_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return metrics_path


def _checkpoint_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        ids.add(str(rec["id"]))
    return ids
