from __future__ import annotations

import pytest

from docintel.config.loader import load_config
from docintel.core.errors import MissingSecretError, StructuredOutputError
from docintel.llm.factory import (
    LangChainCaller,
    ScriptedCaller,
    parse_model_ref,
    require_provider_keys,
    required_env_vars,
)
from docintel.llm.prompts import load_prompts
from docintel.llm.schemas import RouteOut
from docintel.llm.structured import parse_structured, repair_json

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


def test_parse_model_ref_and_required_env() -> None:
    assert parse_model_ref("anthropic:claude-haiku-4-5", "openai") == (
        "anthropic",
        "anthropic:claude-haiku-4-5",
    )
    assert parse_model_ref("gpt-4o-mini", "openai") == ("openai", "openai:gpt-4o-mini")
    cfg = load_config("dev_cpu", repo_root=ROOT)
    assert required_env_vars(cfg) == ["ANTHROPIC_API_KEY"]


def test_missing_key_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "docintel.llm.factory.load_settings",
        lambda: type(
            "S",
            (),
            {"anthropic_api_key": None, "openai_api_key": None, "google_api_key": None},
        )(),
    )
    cfg = load_config("dev_cpu", repo_root=ROOT)
    with pytest.raises(MissingSecretError, match="ANTHROPIC_API_KEY"):
        require_provider_keys(cfg)


def test_scripted_and_json_repair() -> None:
    caller = ScriptedCaller({"router": [RouteOut(route="general", reason="def")]})
    out = caller.structured("router", RouteOut, "q")
    assert out.route == "general"
    repaired = repair_json('```json\n{"route": "out_of_scope", "reason": "x",}\n```')
    parsed = parse_structured(RouteOut, repaired)
    assert parsed.route == "out_of_scope"
    with pytest.raises(StructuredOutputError):
        parse_structured(RouteOut, "not-json")


def test_retry_on_timeout_then_success() -> None:
    class Boom(Exception):
        status_code = 429

    calls = {"n": 0}

    class FakeModel:
        def invoke(self, _messages: object) -> object:
            calls["n"] += 1
            if calls["n"] == 1:
                raise Boom("429 rate limit")
            return RouteOut(route="out_of_scope", reason="x")

        def with_structured_output(self, schema: type[RouteOut]) -> FakeModel:
            _ = schema
            return self

    sleeps: list[float] = []
    caller = LangChainCaller(
        {"router": FakeModel()},
        max_retries=2,
        sleeper=sleeps.append,
        rng=__import__("random").Random(0),
    )
    out = caller.structured("router", RouteOut, "q")
    assert out.route == "out_of_scope"
    assert calls["n"] == 2
    assert sleeps


def test_auth_does_not_retry() -> None:
    class Auth(Exception):
        status_code = 401

    class FakeModel:
        def invoke(self, _messages: object) -> object:
            raise Auth("401 unauthorized")

        def with_structured_output(self, schema: object) -> FakeModel:
            _ = schema
            return self

    caller = LangChainCaller({"router": FakeModel()}, max_retries=3, sleeper=lambda _: None)
    with pytest.raises(Auth):
        caller.structured("router", RouteOut, "q")


def test_structured_falls_back_to_raw_on_parse_failure_only() -> None:
    from docintel.llm.structured import structured

    class Model:
        def __init__(self) -> None:
            self.raw_calls = 0

        def with_structured_output(self, schema: object) -> Model:
            _ = schema
            return _Bad()

        def invoke(self, _messages: object) -> object:
            self.raw_calls += 1
            return type("M", (), {"content": '{"route": "general", "reason": "r"}'})()

    class _Bad:
        def invoke(self, _messages: object) -> object:
            return "garbage"

    model = Model()
    out = structured(model, RouteOut, "q")
    assert out.route == "general"
    assert model.raw_calls == 1


def test_prompt_versions_loaded() -> None:
    bank = load_prompts()
    assert bank.get("generate").text.startswith("Answer using only")
    assert "untrusted" in bank.get("generate").text
    assert "<evidence" in bank.get("grade_batch").text or "evidence" in bank.get("grade_batch").text
    assert bank.versions()["classify"] == "classify-v1"
