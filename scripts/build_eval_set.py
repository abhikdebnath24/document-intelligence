from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from docintel.config.loader import find_repo_root
from docintel.data.corpus import CORPUS_SEED, load_manifest
from docintel.data.evalset import build_eval_items
from docintel.evaluation.gold import dump_qa_set, file_sha256

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build document-disjoint CUAD eval sets")
    parser.add_argument("--manifest", default="data_manifest/corpus_manifest.json")
    parser.add_argument("--dev", default="evals/qa_dev.json")
    parser.add_argument("--test", default="evals/qa_test.json")
    parser.add_argument("--readme", default="evals/README.md")
    parser.add_argument("--seed", type=int, default=CORPUS_SEED)
    args = parser.parse_args()

    root = find_repo_root(ROOT)
    manifest = load_manifest(root / args.manifest)
    eval_docs = [d for d in manifest["documents"] if d.get("split") == "index_and_eval"]
    if not eval_docs:
        raise SystemExit("manifest has no index_and_eval docs; run select_corpus.py first")
    csv_path = root / "data" / "CUAD_v1" / "master_clauses.csv"
    dev, test = build_eval_items(eval_docs, csv_path, seed=args.seed)

    dev_path = root / args.dev
    test_path = root / args.test
    dump_qa_set(dev_path, dev)
    dump_qa_set(test_path, test)
    readme = root / args.readme
    readme.parent.mkdir(parents=True, exist_ok=True)
    buckets = Counter(q.bucket for q in (*dev, *test))
    dev_docs = len({q.doc_stem for q in dev if q.doc_stem})
    test_docs = len({q.doc_stem for q in test if q.doc_stem})
    lines = [
        "# Eval sets v1",
        "",
        "Document-disjoint split.",
        "Tune on `qa_dev.json`. Report `qa_test.json` once per finalist.",
        "",
        f"- `qa_dev.json` sha256 `{file_sha256(dev_path)}` ({len(dev)} items, {dev_docs} docs)",
        f"- `qa_test.json` sha256 `{file_sha256(test_path)}` ({len(test)} items, {test_docs} docs)",
        f"- union buckets: {dict(buckets)}",
        f"- seed `{args.seed}`",
        "",
        "Questions are template-generated from `master_clauses.csv` gold spans.",
        "Edit JSON wording if needed; keep `gold_spans` and `doc_stem`.",
        "",
    ]
    readme.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {dev_path} n={len(dev)}")
    print(f"wrote {test_path} n={len(test)}")
    print(f"wrote {readme}")
    print(f"buckets={dict(buckets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
