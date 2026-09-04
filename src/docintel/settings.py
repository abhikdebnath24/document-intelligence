from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Secrets and process-level flags. Strategy choices live in YAML profiles."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None
    hf_token: str | None = None
    docintel_profile: str = Field(default="dev_cpu")


def _dotenv_paths() -> tuple[str, ...]:
    paths: list[str] = []
    try:
        from docintel.config.loader import find_repo_root

        root_env = find_repo_root() / ".env"
        if root_env.is_file():
            paths.append(str(root_env.resolve()))
    except Exception:
        pass
    cwd_env = Path(".env")
    if cwd_env.is_file():
        resolved = str(cwd_env.resolve())
        if resolved not in paths:
            paths.append(resolved)
    return tuple(paths) or (".env",)


def load_settings() -> Settings:
    settings = Settings(_env_file=_dotenv_paths())  # type: ignore[call-arg]
    token = (settings.hf_token or "").strip()
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    return settings


def hf_token() -> str | None:
    """Export .env HF_TOKEN into the process, then return it. Call before any Hub download."""
    token = (load_settings().hf_token or os.environ.get("HF_TOKEN") or "").strip()
    return token or None
