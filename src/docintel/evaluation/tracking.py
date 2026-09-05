from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from docintel.config import AppConfig, config_hash, index_sig
from docintel.evaluation.experiment import git_sha


@contextmanager
def mlflow_run(
    config: AppConfig,
    *,
    repo_root: Path,
    split: str,
    layer: str,
    extra_params: dict[str, Any] | None = None,
) -> Iterator[str | None]:
    cfg = config.tracking.mlflow
    if not cfg.enabled:
        yield None
        return
    try:
        import mlflow
    except ImportError:
        yield None
        return
    uri = cfg.tracking_uri
    if uri.startswith("file:./"):
        # anchor to the repo, not the shell CWD, so `mlflow ui` finds the same store
        uri = (repo_root / uri[len("file:./") :]).resolve().as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg.experiment)
    params = {
        "profile": config.profile,
        "config_hash": config_hash(config),
        "index_sig": index_sig(config),
        "git_sha": git_sha(repo_root),
        "split": split,
        "layer": layer,
        "chunker": config.ingestion.chunker.name,
        "embedder": config.ingestion.dense_embedder.name,
        "reranker": config.retrieval.reranker.name,
    }
    if extra_params:
        params.update({k: str(v) for k, v in extra_params.items()})
    with mlflow.start_run() as run:
        mlflow.set_tags({"layer": layer, "profile": config.profile})
        mlflow.log_params(params)
        yield run.info.run_id


def log_metrics(metrics: dict[str, float]) -> None:
    try:
        import mlflow
    except ImportError:
        return
    if mlflow.active_run() is None:
        return
    clean = {k: float(v) for k, v in metrics.items() if isinstance(v, int | float)}
    if clean:
        mlflow.log_metrics(clean)


def log_dir(path: Path) -> None:
    try:
        import mlflow
    except ImportError:
        return
    if mlflow.active_run() is None or not path.is_dir():
        return
    mlflow.log_artifacts(str(path))
