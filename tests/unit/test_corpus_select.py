from __future__ import annotations

from docintel.data.corpus import (
    InventoryDoc,
    agreement_group,
    agreement_type_from_folder,
    assign_eval,
    normalize_stem,
    parse_answer_cell,
    parse_clause_cell,
    split_eval_stems,
    stem_family,
    stem_key,
    stratified_take,
)


def _doc(stem: str, atype: str) -> InventoryDoc:
    return InventoryDoc(
        doc_stem=stem,
        rel_path=f"{atype}/{stem}.pdf",
        pdf_path=f"/{stem}.pdf",
        txt_path=f"/{stem}.txt",
        agreement_type=atype,
        group=agreement_group(atype),
        csv_filename=f"{stem}.pdf",
        total_chars=100,
        est_tokens=25,
    )


def test_parse_clause_cell_handles_python_repr_lists() -> None:
    assert parse_clause_cell("['June 8, 2010']") == ["June 8, 2010"]
    assert parse_clause_cell("['A', 'B \"quoted\"']") == ["A", 'B "quoted"']
    assert parse_clause_cell('["json", "list"]') == ["json", "list"]
    assert parse_clause_cell("[]") == []
    assert parse_clause_cell("") == []
    assert parse_clause_cell("plain text") == ["plain text"]
    assert parse_answer_cell("Yes") == "Yes"
    assert parse_answer_cell("['Nevada']") == "Nevada"


def test_normalize_stem_strips_ext_and_case() -> None:
    assert normalize_stem(" Foo.PDF ") == "foo"
    assert stem_key("Foo, Inc. PDF") == stem_key("foo inc pdf")


def test_stem_family_collapses_parts_and_exhibit_suffixes() -> None:
    fam = stem_family("kubient,inc_07_02_2020-ex-10.14-master services agreement_part1")
    assert fam == stem_family("KUBIENT,INC_07_02_2020-EX-10.14-MASTER SERVICES AGREEMENT_Part2.PDF")
    assert stem_family("x franchise agreement1") == stem_family("x franchise agreement3")
    assert stem_family("y consulting agreement(1)") == stem_family("y consulting agreement")
    assert stem_family("acme_2020 supply agreement") != stem_family("other_2020 supply agreement")


def test_assign_eval_skips_stem_families() -> None:
    docs = [_doc(f"solo{i} supply agreement", "Supply") for i in range(6)]
    docs += [
        _doc("twin supply agreement_part1", "Supply"),
        _doc("twin supply agreement_part2", "Supply"),
    ]
    chosen = {d.doc_stem for d in assign_eval(docs, 6, seed=1)}
    assert not any("twin" in s for s in chosen)
    assert len(chosen) == 6


def test_folder_maps_to_canonical_type_and_group() -> None:
    assert agreement_type_from_folder("License_Agreements") == "License"
    assert agreement_type_from_folder("Joint Venture _ Filing") == "Joint Venture"
    assert agreement_group("Distributor") == "core"
    assert agreement_group("License") == "ip"
    assert agreement_group("Agency") == "other"


def test_stratified_take_hits_target_and_keeps_types() -> None:
    types = ("Distributor", "License", "Agency")
    docs = [_doc(f"{atype.lower()}_{i}", atype) for atype in types for i in range(10)]
    selected = stratified_take(docs, target=18, seed=42)
    assert len(selected) == 18
    assert {d.agreement_type for d in selected} == {"Distributor", "License", "Agency"}


def test_stratified_take_tops_up_from_largest_type() -> None:
    docs = (
        [_doc(f"a{i} distributor agreement", "Distributor") for i in range(20)]
        + [_doc(f"b{i} license agreement", "License") for i in range(10)]
        + [_doc(f"c{i} agency agreement", "Agency") for i in range(5)]
    )
    # 0.78*20=16, 0.78*10=8, 0.78*5=4 -> 28; target 30 takes 2 more from Distributor leftover
    selected = stratified_take(docs, target=30, seed=1)
    types = ("Distributor", "License", "Agency")
    counts = {t: sum(1 for d in selected if d.agreement_type == t) for t in types}
    assert sum(counts.values()) == 30
    assert counts["Distributor"] == 18
    assert counts["License"] == 8
    assert counts["Agency"] == 4


def test_split_eval_stems_is_group_balanced_and_deterministic() -> None:
    items = (
        [(f"c{i}", "core") for i in range(24)]
        + [(f"i{i}", "ip") for i in range(16)]
        + [(f"o{i}", "other") for i in range(10)]
    )
    a = split_eval_stems(items, 30, seed=42)
    b = split_eval_stems(items, 30, seed=42)
    assert a == b
    assert len(a) == 30
    by_group = {"core": 0, "ip": 0, "other": 0}
    lookup = dict(items)
    for stem in a:
        by_group[lookup[stem]] += 1
    assert by_group == {"core": 14, "ip": 10, "other": 6}


def test_stratified_take_returns_all_when_under_target() -> None:
    docs = [_doc(f"d{i}", "Distributor") for i in range(5)]
    assert len(stratified_take(docs, target=400, seed=1)) == 5


def test_assign_eval_is_deterministic_and_sized() -> None:
    docs = (
        [_doc(f"c{i} distributor agreement", "Distributor") for i in range(20)]
        + [_doc(f"i{i} license agreement", "License") for i in range(20)]
        + [_doc(f"o{i} agency agreement", "Agency") for i in range(20)]
    )
    a = assign_eval(docs, 50, seed=42)
    b = assign_eval(docs, 50, seed=42)
    assert [d.doc_stem for d in a] == [d.doc_stem for d in b]
    assert len(a) == 50
    assert {d.doc_stem for d in a} <= {d.doc_stem for d in docs}
