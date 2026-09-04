from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from docintel.core.interfaces import BaseQueryTransform
from docintel.data.corpus import normalize_stem

MAX_CANDIDATES = 5

_TYPE_ALIASES: dict[str, str] = {
    "affiliate": "Affiliate",
    "agency": "Agency",
    "co-branding": "Co-Branding",
    "cobranding": "Co-Branding",
    "collaboration": "Collaboration",
    "cooperation": "Collaboration",
    "consulting": "Consulting",
    "development": "Development",
    "distributor": "Distributor",
    "endorsement": "Endorsement",
    "franchise": "Franchise",
    "hosting": "Hosting",
    "ip": "IP",
    "joint venture": "Joint Venture",
    "license": "License",
    "maintenance": "Maintenance",
    "manufacturing": "Manufacturing",
    "marketing": "Marketing",
    "non-compete": "Non-Compete",
    "noncompete": "Non-Compete",
    "outsourcing": "Outsourcing",
    "promotion": "Promotion",
    "reseller": "Reseller",
    "service": "Service",
    "services": "Service",
    "sponsorship": "Sponsorship",
    "strategic alliance": "Strategic Alliance",
    "supply": "Supply",
    "transportation": "Transportation",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def _clause_vocab() -> frozenset[str]:
    # every word of every CUAD category the eval set can ask about, plus the
    # agreement-type aliases: none of these may act as a company-name token
    from docintel.data.evalset import CROSS_PAIRS, SLOT_CATS, YES_CATS

    words: set[str] = set()
    for label, col, _ in [*SLOT_CATS, *YES_CATS]:
        words.update(_TOKEN_RE.findall(f"{label} {col}".lower()))
    for a, b in CROSS_PAIRS:
        words.update(_TOKEN_RE.findall(f"{a} {b}".lower()))
    for alias in _TYPE_ALIASES:
        words.update(_TOKEN_RE.findall(alias))
    return frozenset(words)


_STOP = (
    frozenset(
        {
            "agreement",
            "agreements",
            "clause",
            "clauses",
            "contract",
            "does",
            "include",
            "includes",
            "which",
            "what",
            "whose",
            "jurisdiction",
            "govern",
            "governs",
            "the",
            "and",
            "for",
            "with",
            "from",
            "this",
            "that",
            "under",
            "between",
            "required",
            "measured",
            "appears",
            "grant",
        }
    )
    | _clause_vocab()
)


class IdentityTransform(BaseQueryTransform):
    def transform(self, query: str) -> list[str]:
        return [query]


class FilterExtractor:
    """Regex / catalog matcher. Never reads the gold doc_id.

    doc hint: a question token must equal a token of the catalog stem's company
    segment (text before the first `_` or `-`, the same cut evalset uses to name a
    contract in a question).
    A token that names exactly one doc wins; conflicting tokens -> no doc filter.
    agreement_type: taken from the matched doc's catalog row; regex fallback only when
    no doc matched and the regex type exists in the catalog.
    """

    def __init__(self, catalog: Sequence[dict[str, str]] | None = None) -> None:
        self.catalog = list(catalog or [])
        self._company_tokens: list[tuple[str, frozenset[str], str]] = []
        for row in self.catalog:
            stem = normalize_stem(row.get("doc_stem") or row.get("doc_id") or "")
            # same cut as evalset._display_name; " - " stems have no "_"
            company = stem.split("_")[0].split("-")[0]
            toks = frozenset(t for t in _TOKEN_RE.findall(company) if len(t) >= 4)
            self._company_tokens.append(
                (row.get("doc_id") or stem, toks, row.get("agreement_type", ""))
            )
        self._types = {t for _, _, t in self._company_tokens if t}

    def extract(
        self, question: str, *, use_agreement_type: bool, use_doc_hint: bool
    ) -> dict[str, Any]:
        q = question.lower()
        out: dict[str, Any] = {}
        candidates = self._doc_candidates(q)
        if use_doc_hint and candidates:
            ids = sorted(candidates)
            out["doc_id"] = ids[0] if len(ids) == 1 else ids
        if use_agreement_type:
            types = {t for t in candidates.values() if t}
            if candidates and len(types) == 1:
                out["agreement_type"] = types.pop()
            elif not candidates:
                found = _agreement_type(q)
                if found and (not self._types or found in self._types):
                    out["agreement_type"] = found
        return out

    def _doc_candidates(self, question: str) -> dict[str, str]:
        """doc_id -> agreement_type. One doc when a token is unique; else the smallest
        company-token hit set (<= MAX_CANDIDATES); else empty."""
        tokens = {t for t in _TOKEN_RE.findall(question) if t not in _STOP and len(t) >= 4}
        if not tokens or not self._company_tokens:
            return {}
        hit_sets: list[dict[str, str]] = []
        for tok in tokens:
            hits = {d: t for d, toks, t in self._company_tokens if tok in toks}
            if hits:
                hit_sets.append(hits)
        if not hit_sets:
            return {}
        singles = [h for h in hit_sets if len(h) == 1]
        if singles:
            merged: dict[str, str] = {}
            for h in singles:
                merged.update(h)
            return merged if len(merged) == 1 else {}
        smallest = min(hit_sets, key=len)
        return smallest if len(smallest) <= MAX_CANDIDATES else {}


def _agreement_type(question: str) -> str | None:
    hits: list[str] = []
    for alias, name in sorted(_TYPE_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(alias)}\s+agreement", question):
            if name not in hits:
                hits.append(name)
    if len(hits) == 1:
        return hits[0]
    return None
