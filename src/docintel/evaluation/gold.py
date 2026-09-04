from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from rapidfuzz import fuzz

OMITTED_RE = re.compile(r"<\s*omitted\s*>", re.IGNORECASE)
# join only line-break hyphenation ("agree-\nment"); inline "non-compete" keeps its word gap
HYPHEN_BREAK_RE = re.compile(r"(\w)-[ \t]*\n\s*(\w)")
WS_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s]")
QUOTE_TRANS = str.maketrans(
    {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
    }
)

Bucket = Literal["slot", "yes_span", "no_answer", "cross_ref", "general"]
Split = Literal["dev", "test"]
ExpectedRoute = Literal["general", "corpus_technical", "ambiguous", "out_of_scope"]


class QAItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    doc_stem: str
    agreement_type: str
    category: str
    bucket: Bucket
    question: str
    gold_spans: list[str] = Field(default_factory=list)
    gold_answer: str | None = None
    expected_route: ExpectedRoute = "corpus_technical"
    expected_abstain: bool = False
    split: Split


def normalize_text(text: str) -> str:
    s = text.translate(QUOTE_TRANS).lower()
    s = HYPHEN_BREAK_RE.sub(r"\1\2", s)
    s = PUNCT_RE.sub(" ", s)
    return WS_RE.sub(" ", s).strip()


def gold_fragments(span: str) -> list[str]:
    parts = OMITTED_RE.split(span)
    return [norm for part in parts if (norm := normalize_text(part))]


def all_fragments(spans: list[str]) -> list[str]:
    out: list[str] = []
    for span in spans:
        out.extend(gold_fragments(span))
    return out


class SpanMatcher:
    def __init__(self, threshold: int = 90) -> None:
        if not 0 <= threshold <= 100:
            raise ValueError("threshold must be in 0..100")
        self.threshold = threshold

    def fragment_in_text(self, fragment: str, text: str) -> bool:
        needle = normalize_text(fragment)
        hay = normalize_text(text)
        if not needle:
            return True
        if needle in hay:
            return True
        return bool(fuzz.partial_ratio(needle, hay) >= self.threshold)

    def spans_in_text(self, gold_spans: list[str], text: str) -> bool:
        frags = all_fragments(gold_spans)
        if not frags:
            return False
        return all(self.fragment_in_text(frag, text) for frag in frags)

    def chunk_is_relevant(
        self,
        gold_spans: list[str],
        chunk_text: str,
        *,
        gold_doc_id: str,
        chunk_doc_id: str,
    ) -> bool:
        if gold_doc_id and chunk_doc_id != gold_doc_id:
            return False
        return self.spans_in_text(gold_spans, chunk_text)


def load_qa_set(path: Path) -> list[QAItem]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must be a JSON list")
    return [QAItem.model_validate(item) for item in raw]


def dump_qa_set(path: Path, items: list[QAItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: list[dict[str, Any]] = [item.model_dump() for item in items]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_document_disjoint(dev: list[QAItem], test: list[QAItem]) -> None:
    dev_docs = {item.doc_stem for item in dev if item.doc_stem}
    test_docs = {item.doc_stem for item in test if item.doc_stem}
    overlap = dev_docs & test_docs
    if overlap:
        raise ValueError(f"doc_stem appears in both splits: {sorted(overlap)[:8]}")
