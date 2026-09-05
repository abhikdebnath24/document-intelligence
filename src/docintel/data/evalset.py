from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

from docintel.data.corpus import (
    CORPUS_SEED,
    clause_indexes,
    normalize_stem,
    parse_answer_cell,
    parse_clause_cell,
    split_eval_stems,
    stem_key,
)
from docintel.evaluation.gold import Bucket, ExpectedRoute, QAItem, assert_document_disjoint

Inventory = dict[str, list[tuple[str, list[str], str | None]]]
Chosen = tuple[Bucket, dict[str, str], str, list[str], str | None]
CONTRACT_BUCKETS: tuple[Bucket, ...] = ("slot", "yes_span", "no_answer", "cross_ref")

SLOT_CATS: list[tuple[str, str, str]] = [
    ("Governing Law", "Governing Law", "Governing Law-Answer"),
    ("Parties", "Parties", "Parties-Answer"),
    ("Agreement Date", "Agreement Date", "Agreement Date-Answer"),
    ("Effective Date", "Effective Date", "Effective Date-Answer"),
    ("Expiration Date", "Expiration Date", "Expiration Date-Answer"),
    ("Renewal Term", "Renewal Term", "Renewal Term-Answer"),
    (
        "Notice Period To Terminate Renewal",
        "Notice Period To Terminate Renewal",
        "Notice Period To Terminate Renewal- Answer",
    ),
]
YES_CATS: list[tuple[str, str, str]] = [
    ("Exclusivity", "Exclusivity", "Exclusivity-Answer"),
    ("Audit Rights", "Audit Rights", "Audit Rights-Answer"),
    ("Cap On Liability", "Cap On Liability", "Cap On Liability-Answer"),
    ("Non-Compete", "Non-Compete", "Non-Compete-Answer"),
    ("IP Assignment", "Ip Ownership Assignment", "Ip Ownership Assignment-Answer"),
    ("Insurance", "Insurance", "Insurance-Answer"),
    ("Minimum Commitment", "Minimum Commitment", "Minimum Commitment-Answer"),
    ("License Grant", "License Grant", "License Grant-Answer"),
    ("Anti-Assignment", "Anti-Assignment", "Anti-Assignment-Answer"),
    (
        "Termination For Convenience",
        "Termination For Convenience",
        "Termination For Convenience-Answer",
    ),
]
CROSS_PAIRS: list[tuple[str, str]] = [
    ("Effective Date", "Expiration Date"),
    ("License Grant", "Non-Transferable License"),
]
BUCKET_TARGETS = {
    "slot": 20,
    "yes_span": 24,
    "no_answer": 14,
    "cross_ref": 8,
    "general": 4,
}
GENERAL_ITEMS: list[tuple[str, str, ExpectedRoute]] = [
    ("What is a force majeure clause in a commercial contract?", "general", "general"),
    ("What does a governing-law clause usually decide?", "general", "general"),
    ("What is the weather in Pune today?", "out_of_scope", "out_of_scope"),
    ("Who won the 2019 Cricket World Cup?", "out_of_scope", "out_of_scope"),
]


def _display_name(stem: str, row: dict[str, str], agreement_type: str) -> str:
    doc_name = parse_answer_cell(row.get("Document Name-Answer") or "")
    company = stem.split("_")[0].split("-")[0]
    if company.isupper() or company.islower():
        company = company.title()
    title = doc_name or agreement_type
    return f"{company} {title}".strip()


def _clause(row: dict[str, str], col: str) -> list[str]:
    return parse_clause_cell(row.get(col) or "")


def _answer(row: dict[str, str], col: str) -> str:
    return parse_answer_cell(row.get(col) or "")


def _inventory(row: dict[str, str]) -> Inventory:
    slot: list[tuple[str, list[str], str | None]] = []
    for label, clause_col, ans_col in SLOT_CATS:
        spans = _clause(row, clause_col)
        ans = _answer(row, ans_col)
        if spans and ans and ans.lower() not in {"yes", "no"}:
            slot.append((label, spans, ans))
    yes: list[tuple[str, list[str], str | None]] = []
    no: list[tuple[str, list[str], str | None]] = []
    for label, clause_col, ans_col in YES_CATS:
        spans = _clause(row, clause_col)
        ans = _answer(row, ans_col)
        if spans:
            yes.append((label, spans, "Yes"))
        else:
            no.append((label, [], None))
    xref: list[tuple[str, list[str], str | None]] = []
    for a, b in CROSS_PAIRS:
        sa = _clause(row, a)
        sb = _clause(row, b)
        if sa and sb:
            xref.append((f"{a} + {b}", sa + sb, None))
    return {"slot": slot, "yes_span": yes, "no_answer": no, "cross_ref": xref}


def _question(bucket: str, category: str, display: str) -> str:
    if bucket == "slot":
        if category == "Governing Law":
            return f"Which jurisdiction's law governs the {display}?"
        if category == "Parties":
            return f"Who are the parties to the {display}?"
        if category == "Agreement Date":
            return f"What is the agreement date of the {display}?"
        if category == "Effective Date":
            return f"What is the effective date of the {display}?"
        if category == "Expiration Date":
            return f"What is the expiration date of the {display}?"
        if category == "Renewal Term":
            return f"What is the renewal term in the {display}?"
        if category == "Notice Period To Terminate Renewal":
            return f"What notice period is required to terminate renewal of the {display}?"
        return f"What does the {display} say about {category.lower()}?"
    if bucket == "cross_ref":
        if category.startswith("Effective Date"):
            return (
                f"What is the expiration date of the {display}, "
                "and from which effective date is that term measured?"
            )
        return f"What license grant appears in the {display}, and is that license non-transferable?"
    article = "an" if category[:1].lower() in "aeiou" else "a"
    return f"Does the {display} include {article} {category} clause?"


def build_eval_items(
    eval_docs: list[dict[str, str]],
    csv_path: Path,
    *,
    seed: int = CORPUS_SEED,
    n_dev_docs: int = 30,
) -> tuple[list[QAItem], list[QAItem]]:
    rng = random.Random(seed + 7)
    docs = list(eval_docs)
    docs.sort(key=lambda d: d["doc_stem"])
    rng.shuffle(docs)
    by_stem, by_key = clause_indexes(csv_path)

    def _row_for(stem: str) -> dict[str, str] | None:
        return by_stem.get(normalize_stem(stem)) or by_key.get(stem_key(stem))

    prepared: list[tuple[dict[str, str], Inventory, dict[str, str]]] = []
    for doc in docs:
        row = _row_for(doc["doc_stem"])
        if row is None:
            continue
        prepared.append((doc, _inventory(row), row))

    counts = {k: 0 for k in BUCKET_TARGETS}
    chosen: list[Chosen] = []
    used: set[tuple[str, str, str]] = set()

    def _take(doc: dict[str, str], inv: Inventory, bucket: Bucket) -> bool:
        options = [item for item in inv[bucket] if (doc["doc_stem"], bucket, item[0]) not in used]
        if not options:
            return False
        label, spans, ans = options[0]
        used.add((doc["doc_stem"], bucket, label))
        chosen.append((bucket, doc, label, spans, ans))
        counts[bucket] += 1
        return True

    def _order() -> list[Bucket]:
        return sorted(CONTRACT_BUCKETS, key=lambda b: (counts[b] / BUCKET_TARGETS[b], b))

    # one question per eval doc, filling the most deficient eligible bucket
    for doc, inv, _row in prepared:
        order = _order()
        placed = False
        for bucket in order:
            if counts[bucket] >= BUCKET_TARGETS[bucket]:
                continue
            if _take(doc, inv, bucket):
                placed = True
                break
        if not placed:
            for bucket in order:
                if _take(doc, inv, bucket):
                    break

    n_dev = min(n_dev_docs, len(prepared))
    dev_stems = split_eval_stems(
        [(doc["doc_stem"], doc["group"]) for doc, _inv, _row in prepared],
        n_dev,
        seed,
    )
    # 2 general items per split; aim ~40 / ~30 including those
    contract_targets = {"dev": 38, "test": 28}

    def _split_of(doc: dict[str, str]) -> str:
        return "dev" if doc["doc_stem"] in dev_stems else "test"

    def _contract_count(split: str) -> int:
        return sum(1 for _b, doc, _l, _s, _a in chosen if _split_of(doc) == split)

    changed = True
    while changed:
        changed = False
        for bucket in _order():
            if counts[bucket] >= BUCKET_TARGETS[bucket]:
                continue
            prefer = "test" if _contract_count("test") < contract_targets["test"] else "dev"
            if _contract_count(prefer) >= contract_targets[prefer]:
                prefer = "dev" if prefer == "test" else "test"
            if _contract_count(prefer) >= contract_targets[prefer]:
                continue
            for doc, inv, _row in prepared:
                if _split_of(doc) != prefer:
                    continue
                if counts[bucket] >= BUCKET_TARGETS[bucket]:
                    break
                if _take(doc, inv, bucket):
                    changed = True
                    break

    items: list[QAItem] = []
    for bucket, doc, label, spans, ans in chosen:
        row = _row_for(doc["doc_stem"])
        assert row is not None
        display = _display_name(doc["doc_stem"], row, doc["agreement_type"])
        split: Literal["dev", "test"] = "dev" if doc["doc_stem"] in dev_stems else "test"
        items.append(
            QAItem(
                id="tmp",
                doc_stem=doc["doc_stem"],
                agreement_type=doc["agreement_type"],
                category=label,
                bucket=bucket,
                question=_question(bucket, label, display),
                gold_spans=spans,
                gold_answer=ans,
                expected_route="corpus_technical",
                expected_abstain=bucket == "no_answer",
                split=split,
            )
        )

    for i, (question, category, route) in enumerate(GENERAL_ITEMS):
        general_split: Literal["dev", "test"] = "dev" if i < 2 else "test"
        items.append(
            QAItem(
                id="tmp",
                doc_stem="",
                agreement_type="",
                category=category,
                bucket="general",
                question=question,
                gold_spans=[],
                gold_answer=None,
                expected_route=route,
                expected_abstain=False,
                split=general_split,
            )
        )
        counts["general"] += 1

    items.sort(key=lambda q: (q.split, q.doc_stem, q.bucket, q.category, q.question))
    for i, item in enumerate(items, start=1):
        item.id = f"q_{i:03d}"

    dev = [q for q in items if q.split == "dev"]
    test = [q for q in items if q.split == "test"]
    assert_document_disjoint(dev, test)
    return dev, test
