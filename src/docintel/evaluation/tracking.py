from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from docintel.config import AppConfig, config_hash, index_sig
from docintel.core.logging import get_logger
from docintel.evaluation.experiment import git_sha

log = get_logger(__name__)


def resolve_tracking_uri(uri: str, repo_root: Path) -> str:
    """Anchor relative file/sqlite URIs to repo_root. file: needs MLflow 3.15 opt-out."""
    if uri.startswith("file:"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        rest = uri[len("file:") :]
        if rest.startswith("./"):
            return (repo_root / rest[2:]).resolve().as_uri()
        return uri
    if uri.startswith("sqlite:///"):
        raw = uri[len("sqlite:///") :]
        if raw.startswith(":memory:"):
            return uri
        path = Path(raw)
        if not path.is_absolute():
            path = (repo_root / raw).resolve()
        else:
            path = path.resolve()
        return "sqlite:///" + path.as_posix()
    return uri


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
    uri = resolve_tracking_uri(cfg.tracking_uri, repo_root)
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
    # Store/setup failures must not abort the eval; caller-body exceptions must propagate.
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(cfg.experiment)
        active = mlflow.start_run()
    except Exception as exc:
        log.warning("mlflow_run_skipped", error=str(exc), uri=uri)
        yield None
        return
    with active as run:
        try:
            mlflow.set_tags({"layer": layer, "profile": config.profile})
            mlflow.log_params(params)
        except Exception as exc:
            log.warning("mlflow_params_skipped", error=str(exc))
        yield run.info.run_id


def log_metrics(metrics: dict[str, float]) -> None:
    try:
        import mlflow
    except ImportError:
        return
    if mlflow.active_run() is None:
        return
    clean = {k: float(v) for k, v in metrics.items() if isinstance(v, int | float)}
    if not clean:
        return
    try:
        mlflow.log_metrics(clean)
    except Exception as exc:
        log.warning("mlflow_metrics_skipped", error=str(exc))


def log_dir(path: Path) -> None:
    try:
        import mlflow
    except ImportError:
        return
    if mlflow.active_run() is None or not path.is_dir():
        return
    try:
        mlflow.log_artifacts(str(path))
    except Exception as exc:
        log.warning("mlflow_artifacts_skipped", error=str(exc))
