# Eval sets v1

Document-disjoint split.
Tune on `qa_dev.json`. Report `qa_test.json` once per finalist.

- `qa_dev.json` sha256 `d38d1a012c62d07d90038fb293e24995175b6d06da43d03367502266faea07e8` (40 items, 30 docs)
- `qa_test.json` sha256 `a444f2991582c782ba2adac478dc8d9b1aedd875581a1b312cf4b52d8dee15a5` (30 items, 20 docs)
- union buckets: {'general': 4, 'yes_span': 24, 'slot': 20, 'no_answer': 14, 'cross_ref': 8}
- seed `42`

Questions are template-generated from `master_clauses.csv` gold spans.
Edit JSON wording if needed; keep `gold_spans` and `doc_stem`.
