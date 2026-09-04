from __future__ import annotations

import ast
import csv
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TARGET_DOCS = 400
TARGET_EVAL = 50
CORPUS_SEED = 42
RATIO_LO = 0.97
RATIO_HI = 1.03
CHARS_PER_TOKEN = 4.0

_FOLDER_TO_TYPE = {
    "promotion": "Promotion",
    "non_compete_non_solicit": "Non-Compete",
    "joint venture": "Joint Venture",
    "joint venture _ filing": "Joint Venture",
    "transportation": "Transportation",
    "endorsement": "Endorsement",
    "endorsement agreement": "Endorsement",
    "affiliate_agreements": "Affiliate",
    "affiliate agreement": "Affiliate",
    "development": "Development",
    "co_branding": "Co-Branding",
    "strategic alliance": "Strategic Alliance",
    "franchise": "Franchise",
    "license_agreements": "License",
    "manufacturing": "Manufacturing",
    "reseller": "Reseller",
    "supply": "Supply",
    "sponsorship": "Sponsorship",
    "distributor": "Distributor",
    "hosting": "Hosting",
    "marketing": "Marketing",
    "outsourcing": "Outsourcing",
    "maintenance": "Maintenance",
    "service": "Service",
    "ip": "IP",
    "collaboration": "Collaboration",
    "agency agreements": "Agency",
    "consulting agreements": "Consulting",
}

CORE_TYPES = frozenset({
    "Supply",
    "Manufacturing",
    "Maintenance",
    "Distributor",
    "Outsourcing",
    "Service",
    "Transportation",
})
IP_TYPES = frozenset({
    "License",
    "IP",
    "Joint Venture",
    "Strategic Alliance",
    "Collaboration",
    "Development",
})
OTHER_TYPES = frozenset({
    "Franchise",
    "Reseller",
    "Hosting",
    "Agency",
    "Marketing",
    "Sponsorship",
    "Endorsement",
    "Promotion",
    "Co-Branding",
    "Affiliate",
    "Consulting",
    "Non-Compete",
})
EVAL_QUOTAS = {"core": 24, "ip": 16, "other": 10}


def normalize_stem(name: str) -> str:
    s = Path(name).name.strip().lower()
    for ext in (".pdf", ".txt"):
        if s.endswith(ext):
            s = s[: -len(ext)]
    return " ".join(s.strip().split())


def stem_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize_stem(name))


def agreement_type_from_folder(folder: str) -> str:
    key = folder.strip().lower()
    if key not in _FOLDER_TO_TYPE:
        raise ValueError(f"unmapped agreement-type folder: {folder!r}")
    return _FOLDER_TO_TYPE[key]


def agreement_group(agreement_type: str) -> str:
    if agreement_type in CORE_TYPES:
        return "core"
    if agreement_type in IP_TYPES:
        return "ip"
    if agreement_type in OTHER_TYPES:
        return "other"
    return "other"


def parse_clause_cell(raw: str) -> list[str]:
    # master_clauses.csv stores Python list reprs ("['a', 'b']"), not JSON
    text = (raw or "").strip()
    if not text or text == "[]":
        return []
    val: Any
    try:
        val = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        try:
            val = json.loads(text)
        except json.JSONDecodeError:
            return [text]
    if isinstance(val, list):
        return [str(item).strip() for item in val if str(item).strip()]
    return [str(val).strip()] if str(val).strip() else []


def parse_answer_cell(raw: str) -> str:
    parts = parse_clause_cell(raw)
    if parts:
        return "; ".join(parts)
    return (raw or "").strip()


@dataclass
class InventoryDoc:
    doc_stem: str
    rel_path: str
    pdf_path: str
    txt_path: str | None
    agreement_type: str
    group: str
    csv_filename: str | None
    txt_name: str = ""
    sha256: str = ""
    txt_chars: int = 0
    pdf_chars: int | None = None
    total_chars: int = 0
    est_tokens: int = 0
    n_pages: int | None = None
    pdf_txt_ratio: float | None = None
    ratio_ok: bool | None = None
    split: str = "index"


def stem_family(stem: str) -> str:
    # "franchise agreement1" / "(1)" / "_part2" collapse to one family
    s = normalize_stem(stem)
    s = re.sub(r"[\s_-]*part\s*\d+$", "", s)
    s = re.sub(r"\s*\(\d+\)$", "", s)
    s = re.sub(r"(?<=[a-z])\d+$", "", s)
    return s.strip()


def walk_pdfs(pdf_root: Path) -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    for path in sorted(pdf_root.rglob("*")):
        if path.suffix.lower() != ".pdf" or not path.is_file():
            continue
        atype = agreement_type_from_folder(path.parent.name)
        found.append((path, atype))
    return found


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def clause_indexes(csv_path: Path) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_stem: dict[str, dict[str, str]] = {}
    by_key: dict[str, dict[str, str]] = {}
    for row in rows:
        name = row.get("Filename") or ""
        by_stem[normalize_stem(name)] = row
        by_key.setdefault(stem_key(name), row)
    return by_stem, by_key


def join_clauses(
    pdfs: list[tuple[Path, str]],
    csv_path: Path,
    txt_root: Path,
    pdf_root: Path,
) -> tuple[list[InventoryDoc], list[dict[str, str]]]:
    by_stem, by_key = clause_indexes(csv_path)
    txt_by_stem = {normalize_stem(p.stem): p for p in txt_root.glob("*.txt")}
    txt_by_key = {stem_key(p.stem): p for p in txt_root.glob("*.txt")}

    matched: list[InventoryDoc] = []
    unmatched: list[dict[str, str]] = []
    seen_sha: dict[str, str] = {}
    for pdf_path, atype in pdfs:
        stem = normalize_stem(pdf_path.stem)
        rel = str(pdf_path.relative_to(pdf_root))
        row = by_stem.get(stem) or by_key.get(stem_key(pdf_path.stem))
        txt = txt_by_stem.get(stem) or txt_by_key.get(stem_key(pdf_path.stem))
        reasons: list[str] = []
        if row is None:
            reasons.append("csv")
        if txt is None:
            reasons.append("txt")
        if reasons:
            unmatched.append({"doc_stem": stem, "rel_path": rel, "missing": "+".join(reasons)})
            continue
        assert row is not None
        assert txt is not None
        sha = _sha256_file(pdf_path)
        if sha in seen_sha:
            unmatched.append(
                {"doc_stem": stem, "rel_path": rel, "missing": f"duplicate_of:{seen_sha[sha]}"}
            )
            continue
        seen_sha[sha] = stem
        txt_text = txt.read_text(encoding="utf-8", errors="replace")
        matched.append(
            InventoryDoc(
                doc_stem=stem,
                rel_path=rel,
                pdf_path=str(pdf_path),
                txt_path=str(txt),
                agreement_type=atype,
                group=agreement_group(atype),
                csv_filename=row.get("Filename") or "",
                txt_name=txt.name,
                sha256=sha,
                txt_chars=len(txt_text),
                total_chars=len(txt_text),
                est_tokens=max(1, int(len(txt_text) / CHARS_PER_TOKEN)),
            )
        )
    return matched, unmatched


def stratified_take(
    docs: list[InventoryDoc],
    target: int,
    seed: int,
) -> list[InventoryDoc]:
    if len(docs) <= target:
        return list(docs)
    rng = random.Random(seed)
    by_type: dict[str, list[InventoryDoc]] = {}
    for doc in docs:
        by_type.setdefault(doc.agreement_type, []).append(doc)
    for bucket in by_type.values():
        bucket.sort(key=lambda d: d.doc_stem)
        rng.shuffle(bucket)

    picked: dict[str, list[InventoryDoc]] = {}
    for atype, bucket in by_type.items():
        n = round(0.78 * len(bucket))
        if len(bucket) >= 3:
            n = max(3, n)
        n = min(len(bucket), n)
        picked[atype] = bucket[:n]

    leftover = {t: by_type[t][len(picked[t]) :] for t in by_type}
    selected_n = sum(len(v) for v in picked.values())

    # Plan 3.3: top up or trim from the largest types (most leftover / most picked).
    while selected_n < target:
        candidates = [(len(docs_left), t) for t, docs_left in leftover.items() if docs_left]
        if not candidates:
            break
        _n_left, atype = max(candidates)
        picked[atype].append(leftover[atype].pop(0))
        selected_n += 1

    while selected_n > target:
        candidates = [
            (len(bucket), t)
            for t, bucket in picked.items()
            if len(bucket) > 1
        ]
        if not candidates:
            break
        _n_have, atype = max(candidates)
        leftover[atype].insert(0, picked[atype].pop())
        selected_n -= 1

    selected = [doc for group in picked.values() for doc in group]
    selected.sort(key=lambda d: d.doc_stem)
    return selected


def assign_eval(
    docs: list[InventoryDoc],
    eval_n: int,
    seed: int,
) -> list[InventoryDoc]:
    rng = random.Random(seed + 1)
    # eval docs must be the only member of their stem family, so a question that
    # names the contract cannot be satisfied by a sibling part / exhibit
    families: dict[str, int] = {}
    for doc in docs:
        fam = stem_family(doc.doc_stem)
        families[fam] = families.get(fam, 0) + 1
    by_group: dict[str, list[InventoryDoc]] = {"core": [], "ip": [], "other": []}
    for doc in docs:
        if families[stem_family(doc.doc_stem)] > 1:
            continue
        by_group.setdefault(doc.group, []).append(doc)
    for bucket in by_group.values():
        bucket.sort(key=lambda d: d.doc_stem)
        rng.shuffle(bucket)

    quotas = dict(EVAL_QUOTAS)
    if eval_n != TARGET_EVAL:
        # keep group proportions when the caller shrinks the hold-out
        scale = eval_n / TARGET_EVAL
        quotas = {k: max(1, round(v * scale)) for k, v in EVAL_QUOTAS.items()}

    chosen: list[InventoryDoc] = []
    leftover_slots = 0
    for group in ("core", "ip", "other"):
        want = quotas.get(group, 0) + leftover_slots
        have = by_group.get(group, [])
        take = min(want, len(have))
        chosen.extend(have[:take])
        leftover_slots = want - take

    if len(chosen) < eval_n:
        taken = {d.doc_stem for d in chosen}
        rest = [d for group in by_group.values() for d in group if d.doc_stem not in taken]
        rest.sort(key=lambda d: d.doc_stem)
        rng.shuffle(rest)
        chosen.extend(rest[: eval_n - len(chosen)])
    if len(chosen) > eval_n:
        chosen = chosen[:eval_n]
    return chosen


def split_eval_stems(
    items: list[tuple[str, str]],
    n_dev: int,
    seed: int,
) -> set[str]:
    """Seeded, group-balanced split of (doc_stem, group) pairs. Returns the dev stems."""
    if n_dev <= 0:
        return set()
    if n_dev >= len(items):
        return {stem for stem, _group in items}

    rng = random.Random(seed + 3)
    by_group: dict[str, list[str]] = {}
    for stem, group in items:
        by_group.setdefault(group, []).append(stem)
    for stems in by_group.values():
        stems.sort()
        rng.shuffle(stems)

    n_total = len(items)
    quotas: dict[str, int] = {
        group: min(len(stems), round(len(stems) * n_dev / n_total))
        for group, stems in by_group.items()
    }
    while sum(quotas.values()) > n_dev:
        group = max(quotas, key=lambda g: (quotas[g], g))
        if quotas[group] <= 0:
            break
        quotas[group] -= 1
    while sum(quotas.values()) < n_dev:
        room = [g for g, stems in by_group.items() if quotas[g] < len(stems)]
        if not room:
            break
        group = max(room, key=lambda g: (len(by_group[g]) - quotas[g], g))
        quotas[group] += 1

    dev: set[str] = set()
    for group, stems in by_group.items():
        dev.update(stems[: quotas.get(group, 0)])
    return dev


def _norm_ratio_text(text: str) -> str:
    return "".join(text.lower().split())


def check_pdf_txt(doc: InventoryDoc) -> InventoryDoc:
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("pymupdf is required for the PDF/TXT ratio gate") from exc

    pdf: Any = pymupdf.open(doc.pdf_path)  # type: ignore[no-untyped-call]
    try:
        n_pages = int(pdf.page_count)
        pages = [str(pdf.load_page(i).get_text()) for i in range(n_pages)]
    finally:
        pdf.close()
    pdf_text = "\n".join(pages)
    doc.n_pages = n_pages
    doc.pdf_chars = len(pdf_text)
    txt = Path(doc.txt_path).read_text(encoding="utf-8", errors="replace") if doc.txt_path else ""
    denom = len(_norm_ratio_text(txt))
    numer = len(_norm_ratio_text(pdf_text))
    if denom == 0 or numer == 0 or n_pages < 1:
        doc.pdf_txt_ratio = None
        doc.ratio_ok = False
        return doc
    doc.pdf_txt_ratio = numer / denom
    doc.ratio_ok = RATIO_LO <= doc.pdf_txt_ratio <= RATIO_HI
    return doc


def apply_eval_ratio_gate(
    selected: list[InventoryDoc],
    eval_docs: list[InventoryDoc],
) -> tuple[list[InventoryDoc], list[str]]:
    eval_stems = {d.doc_stem for d in eval_docs}
    index_pool = [d for d in selected if d.doc_stem not in eval_stems]
    notes: list[str] = []
    final_eval: list[InventoryDoc] = []

    def _same_type_replacements(failed: InventoryDoc) -> list[InventoryDoc]:
        same_type = [d for d in index_pool if d.agreement_type == failed.agreement_type]
        same_group = [d for d in index_pool if d.group == failed.group]
        return same_type + [d for d in same_group if d not in same_type] + [
            d for d in index_pool if d not in same_type and d not in same_group
        ]

    for doc in eval_docs:
        check_pdf_txt(doc)
        if doc.ratio_ok:
            final_eval.append(doc)
            continue
        notes.append(f"ratio_fail {doc.doc_stem} ratio={doc.pdf_txt_ratio} pages={doc.n_pages}")
        replacement: InventoryDoc | None = None
        for cand in _same_type_replacements(doc):
            check_pdf_txt(cand)
            if cand.ratio_ok:
                replacement = cand
                index_pool.remove(cand)
                break
        if replacement is None:
            notes.append(f"no_replacement {doc.doc_stem}")
            continue
        notes.append(f"replaced {doc.doc_stem} -> {replacement.doc_stem}")
        final_eval.append(replacement)
    return final_eval, notes


def select_corpus(
    pdf_root: Path,
    txt_root: Path,
    csv_path: Path,
    *,
    target: int = TARGET_DOCS,
    eval_n: int = TARGET_EVAL,
    seed: int = CORPUS_SEED,
    apply_ratio: bool = True,
) -> dict[str, Any]:
    pdfs = walk_pdfs(pdf_root)
    matched, unmatched = join_clauses(pdfs, csv_path, txt_root, pdf_root)
    selected = stratified_take(matched, target, seed)
    eval_n = min(eval_n, len(selected))
    eval_docs = assign_eval(selected, eval_n, seed)
    ratio_notes: list[str] = []
    if apply_ratio:
        eval_docs, ratio_notes = apply_eval_ratio_gate(selected, eval_docs)
    eval_stems = {d.doc_stem for d in eval_docs}
    for doc in selected:
        doc.split = "index_and_eval" if doc.doc_stem in eval_stems else "index"

    types = sorted({d.agreement_type for d in selected})
    drop = {"pdf_path", "txt_path"}
    documents = [{k: v for k, v in asdict(d).items() if k not in drop} for d in selected]
    return {
        "version": 1,
        "seed": seed,
        "target_docs": target,
        "target_eval": eval_n,
        "n_available_pdfs": len(pdfs),
        "n_matched": len(matched),
        "n_selected": len(selected),
        "n_eval": len(eval_docs),
        "agreement_types": types,
        "unmatched": unmatched,
        "ratio_notes": ratio_notes,
        "documents": documents,
    }


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a JSON object")
    return raw
