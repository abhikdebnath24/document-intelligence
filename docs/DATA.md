# Data

## Source

CUAD v1 (Contract Understanding Atticus Dataset). CC BY 4.0. SEC EDGAR commercial contracts with 41 lawyer-labelled clause types.

- Zenodo: https://zenodo.org/records/4595826 (`CUAD_v1.zip`, 105.9 MB)
- Local (gitignored): `data/CUAD_v1/`
- Indexed: `full_contract_pdf/Part_{I,II,III}/<Agreement Type>/*.pdf`
- Validation oracle only: `full_contract_txt/*.txt`
- Gold: `master_clauses.csv` (not `CUAD_v1.json` `answer_start`)

## Subset rule

`scripts/select_corpus.py` walks PDFs, maps the parent folder to a canonical agreement type, joins `master_clauses.csv` by normalized stem, then stratified-samples toward 400 docs (78% per type, min 3 when the type has >= 3). Shortfall and overflow are taken from the largest types, not a global leftover shuffle. 50 of those are `index_and_eval`, weighted 24 / 16 / 10 across core / IP / other. The 30/20 `qa_dev` / `qa_test` cut is a seeded shuffle inside each group so both files keep that mix.

Local tree: 510 PDFs (311 named `.PDF`, 199 `.pdf`), 510 TXT. `select_corpus.py` walks all 510: 3 fail the CSV/TXT join, 1 is a byte-identical duplicate (`ADUROBIOTECH ... CONSULTING AGREEMENT` vs `(1)`), so 506 are joinable and 400 are selected per the A3 rule. The 106 not selected are decoys left out on purpose; `--target 510` indexes everything. `corpus.limit_docs` shrinks smoke runs (`dev_cpu` = 20).

Eval docs are the only member of their stem family (`_part1/_part2`, `agreement1/3`, `(1)`), so a question that names the contract has one valid `doc_id`.

## Split

| File | Role |
|------|------|
| `data_manifest/corpus_manifest.json` | selected docs, sha256, type, group, `index` / `index_and_eval` |
| `evals/qa_dev.json` | tuning (~30 eval docs) |
| `evals/qa_test.json` | reported once per finalist (~20 eval docs) |

Split is by document. A test question never shares `doc_stem` with a dev question. Shas live in `evals/README.md`.

## Known quirks

- PDF folder is `Part_* / <type>`, not a flat type tree. Type aliases (`License_Agreements`, `Joint Venture _ Filing`) are mapped in `docintel.data.corpus`.
- CSV `Filename` is joined with `strip().lower()` plus an alnum key for punctuation drift.
- Clause cells are Python list reprs (`"['June 8, 2010']"`), not JSON; parse with `ast.literal_eval`. Absent labels are `[]`. Yes/No categories store `Yes` / `No` in the `*-Answer` column.
- `Parties` spans are many short fragments (defined terms like `"THI"`). The matcher requires every fragment; short ones match almost any chunk, so Parties recall is optimistic. Report it separately if it skews a bucket.
- Gold spans may contain `<omitted>`. `SpanMatcher` splits on it and requires every fragment.
- `answer_start` in `CUAD_v1.json` is TXT-offset and does not align to PDF text. Do not use it.
- TXT is the length oracle. Eval docs must pass `len(norm(pdf)) / len(norm(txt))` in `[0.97, 1.03]` and have at least one non-empty PDF page. Failures are replaced from the same agreement type.
- Three PDFs do not join CSV (one also has no TXT twin) and one is a duplicate. All four are in `unmatched` with a reason and are not indexed. `rglob("*.pdf")` misses uppercase `.PDF`; walk by suffix instead.
- Manifest `documents[].rel_path` is relative to `corpus.pdf_root`; `txt_name` is the file under `corpus.txt_root` (case can differ from the PDF stem).
