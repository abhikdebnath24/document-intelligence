from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from docintel.config.schema import AppConfig, NamedStrategy
from docintel.core.errors import ConfigError

_PATH_KEYS = frozenset(
    {
        "manifest",
        "pdf_root",
        "txt_root",
        "path",
        "url",
        "db_url",
        "tracking_uri",
        "jsonl_dir",
        "qa_dev",
        "qa_test",
        "finalists_file",
        "results_root",
        "api_url",
        "endpoint",
        "source_path",
    }
)
_SECRET_EXACT = frozenset({"password", "secret", "token", "api_key", "api_key_env"})
_SECRET_SUFFIXES = ("_key", "_secret", "_password")
_ENV_PREFIX = "DOCINTEL__"


def find_repo_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate
    raise ConfigError("could not find repo root (pyproject.toml + configs/)")


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"missing config file: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must be a mapping")
    return raw


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(sample: Any, raw: str) -> Any:
    if isinstance(sample, bool):
        lowered = raw.lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ConfigError(f"cannot coerce {raw!r} to bool")
    if isinstance(sample, int) and not isinstance(sample, bool):
        return int(raw)
    if isinstance(sample, float):
        return float(raw)
    if isinstance(sample, list):
        return json.loads(raw)
    return raw


def apply_env_overrides(
    data: dict[str, Any], environ: dict[str, str] | None = None
) -> dict[str, Any]:
    env = environ if environ is not None else os.environ
    out: dict[str, Any] = json.loads(json.dumps(data))
    for key, raw in env.items():
        if not key.startswith(_ENV_PREFIX):
            continue
        parts = [p.lower() for p in key[len(_ENV_PREFIX) :].split("__") if p]
        if not parts:
            continue
        cursor: Any = out
        for part in parts[:-1]:
            if not isinstance(cursor, dict) or part not in cursor:
                raise ConfigError(f"env {key} does not match any config path")
            cursor = cursor[part]
        leaf = parts[-1]
        if not isinstance(cursor, dict) or leaf not in cursor:
            raise ConfigError(f"env {key} does not match any config path")
        cursor[leaf] = _coerce(cursor[leaf], raw)
    return out


def _is_secret_key(name: str) -> bool:
    lowered = name.lower()
    return lowered in _SECRET_EXACT or lowered.endswith(_SECRET_SUFFIXES)


def _strip_for_hash(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: _strip_for_hash(v)
            for k, v in sorted(value.items())
            if not (k in _PATH_KEYS or _is_secret_key(k))
        }
    if isinstance(value, list):
        return [_strip_for_hash(item) for item in value]
    return value


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: AppConfig) -> str:
    dumped = config.model_dump(mode="json")
    dumped.pop("profile", None)
    stripped = _strip_for_hash(dumped)
    return hashlib.sha256(canonical_json(stripped).encode("utf-8")).hexdigest()


# Runtime knobs that do not change the produced vectors. Excluded so a batch-size
# tweak on the GPU box does not force a full re-ingest.
_INDEX_RUNTIME_KEYS = frozenset({"device", "batch_size"})


def _strategy_for_sig(strategy: NamedStrategy) -> dict[str, Any]:
    dumped = strategy.model_dump(mode="json")
    dumped["params"] = {
        k: v for k, v in dumped.get("params", {}).items() if k not in _INDEX_RUNTIME_KEYS
    }
    return dumped


def index_sig(config: AppConfig) -> str:
    ing = config.ingestion
    payload = {
        "loader": _strategy_for_sig(ing.loader),
        "chunker": _strategy_for_sig(ing.chunker),
        "dense_embedder": _strategy_for_sig(ing.dense_embedder),
        "sparse_encoder": _strategy_for_sig(ing.sparse_encoder),
        "pipeline_version": ing.pipeline_version,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def load_config(
    profile: str = "dev_cpu",
    *,
    repo_root: Path | None = None,
    environ: dict[str, str] | None = None,
) -> AppConfig:
    root = repo_root or find_repo_root()
    base = _read_yaml(root / "configs" / "base.yaml")
    merged = base
    if profile and profile != "base":
        merged = deep_merge(base, _read_yaml(root / "configs" / "profiles" / f"{profile}.yaml"))
    merged = apply_env_overrides(merged, environ)
    try:
        cfg = AppConfig.model_validate({**merged, "profile": profile})
    except Exception as exc:
        raise ConfigError(f"invalid config for profile {profile!r}: {exc}") from exc
    return cfg
