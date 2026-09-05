from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from docintel.config import AppConfig, config_hash, index_sig
from docintel.config.loader import find_repo_root
from docintel.evaluation.custom_metrics import (
    agreement_markdown,
    custom_metrics,
    framework_agreement,
    top_misses,
)
from docintel.evaluation.experiment import FinalistGateError, git_sha, load_finalists
from docintel.evaluation.frameworks.base import (
    EvalResult,
    record_from_ask,
    samples_from_records,
)
from docintel.evaluation.gold import file_sha256, load_qa_set
from docintel.evaluation.tracking import log_dir, log_metrics, mlflow_run
from docintel.service.container import Container
from docintel.service.query_service import QueryService
from docintel.settings import load_settings


def run_generation_eval(
    config: AppConfig,
    *,
    split: str,
    repo_root: Path | None = None,
    frameworks: list[str] | None = None,
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
    c_hash = config_hash(config)
    run_id = f"{config.profile}_{split}_{c_hash[:12]}_L2"
    out_dir = root / config.evaluation.results_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "generation_outputs.jsonl"
    records = _ask_all(config, items, ckpt, root)
    samples = samples_from_records(
        items, records, match_threshold=config.evaluation.span_match_threshold
    )
    custom = custom_metrics(samples)
    (out_dir / "custom_metrics.json").write_text(
        json.dumps(custom, indent=2) + "\n", encoding="utf-8"
    )
    # [] means custom-only; None means "whatever the profile says"
    names = (
        frameworks if frameworks is not None else [f.value for f in config.evaluation.frameworks]
    )
    ragas_res = deepeval_res = None
    if "ragas" in names:
        ragas_res = _eval_ragas(config, samples)
        (out_dir / "generation_ragas.json").write_text(
            ragas_res.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    if "deepeval" in names:
        deepeval_res = _eval_deepeval(config, samples)
        (out_dir / "generation_deepeval.json").write_text(
            deepeval_res.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    if ragas_res and deepeval_res:
        agree = framework_agreement(ragas_res, deepeval_res)
        (out_dir / "framework_agreement.md").write_text(agreement_markdown(agree), encoding="utf-8")
    l1 = _latest_l1(root / config.evaluation.results_root, config.profile, split, c_hash)
    if l1 is not None:
        misses = top_misses(l1)
        (out_dir / "error_analysis.json").write_text(
            json.dumps(misses, indent=2) + "\n", encoding="utf-8"
        )
    summary = {
        "profile": config.profile,
        "split": split,
        "layer": "L2",
        "config_hash": c_hash,
        "index_sig": index_sig(config),
        "qa_sha": file_sha256(qa_path),
        "git_sha": git_sha(root),
        "n": len(samples),
        "custom": custom,
        "ragas": ragas_res.overall if ragas_res else {},
        "deepeval": deepeval_res.overall if deepeval_res else {},
    }
    (out_dir / "summary.md").write_text(_summary_md(summary), encoding="utf-8")
    (out_dir / "config_hash.txt").write_text(c_hash + "\n", encoding="utf-8")
    metrics = {**{f"custom_{k}": v for k, v in custom.items()}}
    if ragas_res:
        metrics.update({f"ragas_{k}": v for k, v in ragas_res.overall.items()})
    if deepeval_res:
        metrics.update({f"deepeval_{k}": v for k, v in deepeval_res.overall.items()})
    with mlflow_run(config, repo_root=root, split=split, layer="L2") as mlf_id:
        log_metrics(metrics)
        log_dir(out_dir)
        summary["mlflow_run_id"] = mlf_id
        (out_dir / "generation_metrics.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    return out_dir / "generation_metrics.json"


def _ask_all(config: AppConfig, items: list[Any], ckpt: Path, root: Path) -> list[dict[str, Any]]:
    done = _checkpoint_ids(ckpt)
    rows: list[dict[str, Any]] = []
    if ckpt.is_file():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    pending = [item for item in items if item.id not in done]
    if not pending:
        return rows
    svc = QueryService(Container(config, repo_root=root))
    try:
        with ckpt.open("a", encoding="utf-8") as fh:
            for item in pending:
                answer, log = svc.ask(item.question)
                rec = record_from_ask(item, answer, log)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                rows.append(rec)
    finally:
        svc.close()
    return rows


def _eval_ragas(config: AppConfig, samples: Any) -> EvalResult:
    from docintel.evaluation.frameworks.ragas_adapter import RagasEvaluator

    return RagasEvaluator(config).evaluate(samples)


def _eval_deepeval(config: AppConfig, samples: Any) -> EvalResult:
    from docintel.evaluation.frameworks.deepeval_adapter import DeepEvalEvaluator

    return DeepEvalEvaluator(config).evaluate(samples)


def _latest_l1(
    results_root: Path, profile: str, split: str, c_hash: str
) -> list[dict[str, Any]] | None:
    """L1 per-question rows for this exact config; else the newest same-profile run."""
    exact = results_root / f"{profile}_{split}_{c_hash[:12]}" / "per_question.jsonl"
    if exact.is_file():
        ckpt = exact
    else:
        if not results_root.is_dir():
            return None
        dirs = [
            p
            for p in results_root.iterdir()
            if p.is_dir()
            and p.name.startswith(f"{profile}_{split}_")
            and not p.name.endswith(("_L2", "_scoped"))
        ]
        if not dirs:
            return None
        ckpt = max(dirs, key=lambda p: p.stat().st_mtime) / "per_question.jsonl"
        if not ckpt.is_file():
            return None
    return [
        json.loads(line) for line in ckpt.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _checkpoint_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(str(json.loads(line)["id"]))
    return ids


def _summary_md(payload: dict[str, Any]) -> str:
    custom = payload.get("custom") or {}
    lines = [
        f"# L2 {payload['profile']} {payload['split']}",
        "",
        f"config_hash={payload['config_hash']}",
        f"index_sig={payload['index_sig']}",
        f"qa_sha={payload['qa_sha']}",
        f"n={payload['n']}",
        "",
        f"faithfulness_ragas={(payload.get('ragas') or {}).get('faithfulness', 'n/a')}",
        f"route_accuracy={custom.get('route_accuracy', 'n/a')}",
        f"abstention_recall={custom.get('abstention_recall', 'n/a')}",
        f"latency_p50_ms={custom.get('latency_p50_ms', 'n/a')}",
        f"llm_calls_per_query={custom.get('llm_calls_per_query', 'n/a')}",
        "",
    ]
    return "\n".join(lines)
