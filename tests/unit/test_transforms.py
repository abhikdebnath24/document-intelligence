from __future__ import annotations

import json
from pathlib import Path

from docintel.data.corpus import normalize_stem
from docintel.retrieval.transforms import FilterExtractor

ROOT = Path(__file__).resolve().parents[2]


def test_doc_hint_uses_company_segment_and_catalog_type() -> None:
    catalog = [
        {
            "doc_id": "abilityinc_06_15_2020-ex-4.25-services agreement",
            "doc_stem": "abilityinc_06_15_2020-ex-4.25-services agreement",
            "agreement_type": "Service",
        },
        {
            "doc_id": "otherco_01_01_2020-ex-1-non-compete agreement",
            "doc_stem": "otherco_01_01_2020-ex-1-non-compete agreement",
            "agreement_type": "Non-Compete",
        },
    ]
    ext = FilterExtractor(catalog)
    got = ext.extract(
        "Does the Abilityinc Services Agreement include an IP Assignment clause?",
        use_agreement_type=True,
        use_doc_hint=True,
    )
    assert got == {
        "doc_id": "abilityinc_06_15_2020-ex-4.25-services agreement",
        "agreement_type": "Service",
    }
    # clause words after the company segment must not pin a doc
    none = ext.extract(
        "Does the Ceres,Inc COLLABORATION AGREEMENT include a Non-Compete clause?",
        use_agreement_type=True,
        use_doc_hint=True,
    )
    assert "doc_id" not in none
    # regex type only when it exists in the catalog
    assert none == {}


def test_ambiguous_company_becomes_candidate_list_not_type_guess() -> None:
    catalog = [
        {"doc_id": "acme_1-service", "doc_stem": "acme_1-service", "agreement_type": "Service"},
        {"doc_id": "acme_2-license", "doc_stem": "acme_2-license", "agreement_type": "License"},
    ]
    got = FilterExtractor(catalog).extract(
        "Does the Acme Software License Agreement include exclusivity?",
        use_agreement_type=True,
        use_doc_hint=True,
    )
    # title says License but the doc may be filed elsewhere: restrict by doc set, not type
    assert got == {"doc_id": ["acme_1-service", "acme_2-license"]}


def test_extractor_never_pins_wrong_doc_on_eval_questions() -> None:
    manifest = json.loads(
        (ROOT / "data_manifest" / "corpus_manifest.json").read_text(encoding="utf-8")
    )
    catalog = [
        {
            "doc_id": normalize_stem(d["doc_stem"]),
            "doc_stem": d["doc_stem"],
            "agreement_type": d["agreement_type"],
        }
        for d in manifest["documents"]
    ]
    ext = FilterExtractor(catalog)
    items = json.loads((ROOT / "evals" / "qa_dev.json").read_text(encoding="utf-8"))
    items += json.loads((ROOT / "evals" / "qa_test.json").read_text(encoding="utf-8"))
    scored = [i for i in items if i["doc_stem"]]
    found = 0
    for item in scored:
        got = ext.extract(item["question"], use_agreement_type=True, use_doc_hint=True)
        gold = normalize_stem(item["doc_stem"])
        doc = got.get("doc_id")
        if isinstance(doc, list):
            assert gold in doc, item["question"]
        elif doc is not None:
            assert doc == gold, item["question"]
            found += 1
        if "agreement_type" in got:
            # a wrong hard type filter zeroes the question; never allowed on eval data
            assert got["agreement_type"] == item["agreement_type"], item["question"]
    # the hint must be useful, not just safe
    assert found >= len(scored) // 2, f"doc hint found on only {found}/{len(scored)}"
