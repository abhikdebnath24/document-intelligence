from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from docintel.core.errors import StructuredOutputError

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_TRAIL_COMMA = re.compile(r",\s*([}\]])")


def repair_json(text: str) -> str:
    raw = text.strip()
    raw = _FENCE.sub("", raw).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    return _TRAIL_COMMA.sub(r"\1", raw)


def parse_structured[T: BaseModel](schema: type[T], payload: object) -> T:
    if isinstance(payload, schema):
        return payload
    if isinstance(payload, BaseModel):
        return schema.model_validate(payload.model_dump())
    if isinstance(payload, dict):
        return schema.model_validate(payload)
    if not isinstance(payload, str):
        raise StructuredOutputError(f"cannot parse {type(payload).__name__} as {schema.__name__}")
    try:
        return schema.model_validate_json(payload)
    except (ValidationError, json.JSONDecodeError):
        repaired = repair_json(payload)
        try:
            return schema.model_validate_json(repaired)
        except (ValidationError, json.JSONDecodeError) as exc:
            raise StructuredOutputError(f"malformed {schema.__name__}: {exc}") from exc


def structured[T: BaseModel](model: Any, schema: type[T], prompt: str, *, system: str = "") -> T:
    """with_structured_output first; JSON-repair fallback on parse failure."""
    messages = _messages(prompt, system)
    binder = getattr(model, "with_structured_output", None)
    if binder is not None:
        # parse failures fall through to a raw call + JSON repair; API errors propagate
        try:
            return parse_structured(schema, binder(schema).invoke(messages))
        except _PARSE_ERRORS:
            pass
    raw = model.invoke(messages)
    text = _content(raw)
    return parse_structured(schema, text)


def _parse_errors() -> tuple[type[Exception], ...]:
    base: tuple[type[Exception], ...] = (StructuredOutputError, ValidationError)
    try:
        from langchain_core.exceptions import OutputParserException
    except ImportError:
        return base
    return (*base, OutputParserException)


_PARSE_ERRORS = _parse_errors()


def _messages(prompt: str, system: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if system:
        out.append({"role": "system", "content": system})
    out.append({"role": "user", "content": prompt})
    return out


def _content(raw: object) -> str:
    content = getattr(raw, "content", raw)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return str(content)
