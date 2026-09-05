from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_FRONT = re.compile(r"^---\s*\nversion:\s*(\S+)\s*\n---\s*\n", re.MULTILINE)

_FILES = {
    "classify": "classify.md",
    "grade_batch": "grade_batch.md",
    "grade_one": "grade_one.md",
    "rewrite": "rewrite.md",
    "generate": "generate.md",
    "generate_strict": "generate_strict.md",
    "verify": "verify.md",
    "general": "general.md",
    "clarify": "clarify.md",
}


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    text: str


class PromptBank:
    def __init__(self, items: dict[str, Prompt]) -> None:
        self._items = items

    def get(self, name: str) -> Prompt:
        return self._items[name]

    def versions(self) -> dict[str, str]:
        return {name: item.version for name, item in self._items.items()}


def load_prompts() -> PromptBank:
    root = Path(__file__).resolve().parent
    items: dict[str, Prompt] = {}
    for name, filename in _FILES.items():
        raw = (root / filename).read_text(encoding="utf-8")
        match = _FRONT.match(raw)
        if match:
            version = match.group(1)
            body = raw[match.end() :]
        else:
            version = f"{name}-v0"
            body = raw
        items[name] = Prompt(name=name, version=version, text=body.strip())
    return PromptBank(items)
