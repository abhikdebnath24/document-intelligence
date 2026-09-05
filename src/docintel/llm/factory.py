from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from docintel.config import AppConfig
from docintel.core.errors import MissingSecretError, StructuredOutputError
from docintel.llm.structured import parse_structured, structured
from docintel.settings import load_settings

T = TypeVar("T", bound=BaseModel)

PROVIDER_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
}

_AUTH_HINTS = ("401", "403", "authentication", "invalid api key", "unauthorized", "permission")
_RETRY_HINTS = ("429", "500", "502", "503", "504", "timeout", "timed out", "temporarily")


class StructuredCaller(Protocol):
    def structured(self, role: str, schema: type[T], prompt: str, *, system: str = "") -> T: ...

    def text(self, role: str, prompt: str, *, system: str = "") -> str: ...


class ScriptedCaller:
    """Test double. Queue per role: schema instances, dicts, strings, or exceptions."""

    def __init__(self, scripts: dict[str, list[object]] | None = None) -> None:
        self.scripts = {key: list(vals) for key, vals in (scripts or {}).items()}

    def structured(self, role: str, schema: type[T], prompt: str, *, system: str = "") -> T:
        _ = prompt, system
        return parse_structured(schema, self._next(role))

    def text(self, role: str, prompt: str, *, system: str = "") -> str:
        _ = prompt, system
        item = self._next(role)
        return item if isinstance(item, str) else str(item)

    def _next(self, role: str) -> object:
        queue = self.scripts.get(role)
        if not queue:
            raise StructuredOutputError(f"scripted caller exhausted for role {role!r}")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class LangChainCaller:
    def __init__(
        self,
        models: dict[str, Any],
        *,
        max_retries: int = 3,
        sleeper: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.models = models
        self.max_retries = max(0, max_retries)
        self.sleeper = sleeper
        self.rng = rng or random.Random()

    def structured(self, role: str, schema: type[T], prompt: str, *, system: str = "") -> T:
        return self._retry(lambda: structured(self._model(role), schema, prompt, system=system))

    def text(self, role: str, prompt: str, *, system: str = "") -> str:
        from docintel.llm.structured import _content, _messages

        model = self._model(role)
        return self._retry(lambda: _content(model.invoke(_messages(prompt, system))))

    def _model(self, role: str) -> Any:
        if role not in self.models:
            raise KeyError(f"no chat model for role {role!r}")
        return self.models[role]

    def _retry[R](self, fn: Callable[[], R]) -> R:
        last: Exception | None = None
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                return fn()
            except MissingSecretError:
                raise
            except Exception as exc:
                last = exc
                if _is_auth(exc) or attempt == attempts - 1 or not _is_retryable(exc):
                    raise
                self.sleeper(self.rng.uniform(0.2, 0.8) * (2**attempt))
        assert last is not None
        raise last


def parse_model_ref(ref: str, default_provider: str) -> tuple[str, str]:
    text = ref.strip()
    if ":" in text:
        provider, model = text.split(":", 1)
        return provider, f"{provider}:{model}"
    return default_provider, f"{default_provider}:{text}"


def required_env_vars(config: AppConfig) -> list[str]:
    names: list[str] = []
    for role in config.llm.roles.values():
        provider, _ = parse_model_ref(role.model, config.llm.default_provider)
        env = PROVIDER_ENV.get(provider)
        if env and env not in names:
            names.append(env)
    return names


def export_provider_keys() -> None:
    settings = load_settings()
    pairs = (
        ("ANTHROPIC_API_KEY", settings.anthropic_api_key),
        ("OPENAI_API_KEY", settings.openai_api_key),
        ("GOOGLE_API_KEY", settings.google_api_key),
    )
    for key, value in pairs:
        if value:
            os.environ.setdefault(key, value)


def require_provider_keys(config: AppConfig) -> None:
    export_provider_keys()
    for env in required_env_vars(config):
        if not (os.environ.get(env) or "").strip():
            raise MissingSecretError(env)


def build_chat_model(role: str, config: AppConfig) -> Any:
    require_provider_keys(config)
    spec = config.llm.roles[role]
    provider, model_id = parse_model_ref(spec.model, config.llm.default_provider)
    _ = provider
    from langchain.chat_models import init_chat_model

    kwargs: dict[str, Any] = {
        "temperature": spec.temperature,
        "timeout": config.llm.timeout_s,
        "max_retries": 0,
    }
    if spec.max_tokens is not None:
        kwargs["max_tokens"] = spec.max_tokens
    return init_chat_model(model_id, **kwargs)


def build_caller(config: AppConfig) -> LangChainCaller:
    models = {role: build_chat_model(role, config) for role in config.llm.roles}
    return LangChainCaller(models, max_retries=config.llm.max_retries)


def _is_auth(exc: BaseException) -> bool:
    status = _status(exc)
    if status in {401, 403}:
        return True
    text = str(exc).lower()
    return any(hint in text for hint in _AUTH_HINTS if hint not in {"401", "403", "429"})


def _is_retryable(exc: BaseException) -> bool:
    if _is_auth(exc):
        return False
    if isinstance(exc, StructuredOutputError):
        return True
    status = _status(exc)
    if status in {429, 500, 502, 503, 504}:
        return True
    text = str(exc).lower()
    return any(hint in text for hint in _RETRY_HINTS)


def _status(exc: BaseException) -> int | None:
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None
