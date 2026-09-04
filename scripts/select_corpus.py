from __future__ import annotations

import argparse
from pathlib import Path

from docintel.config.loader import find_repo_root
from docintel.data.corpus import (
    CORPUS_SEED,
    TARGET_DOCS,
    TARGET_EVAL,
    select_corpus,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Stratified CUAD corpus manifest")
    parser.add_argument("--out", default="data_manifest/corpus_manifest.json")
    parser.add_argument("--seed", type=int, default=CORPUS_SEED)
    parser.add_argument("--target", type=int, default=TARGET_DOCS)
    parser.add_argument("--eval-n", type=int, default=TARGET_EVAL)
    parser.add_argument("--skip-ratio", action="store_true")
    args = parser.parse_args()

    root = find_repo_root(ROOT)
    pdf_root = root / "data" / "CUAD_v1" / "full_contract_pdf"
    txt_root = root / "data" / "CUAD_v1" / "full_contract_txt"
    csv_path = root / "data" / "CUAD_v1" / "master_clauses.csv"
    if not pdf_root.is_dir():
        raise SystemExit(f"missing PDF root: {pdf_root}")
    if not csv_path.is_file():
        raise SystemExit(f"missing CSV: {csv_path}")

    payload = select_corpus(
        pdf_root,
        txt_root,
        csv_path,
        target=args.target,
        eval_n=args.eval_n,
        seed=args.seed,
        apply_ratio=not args.skip_ratio,
    )
    out = root / args.out
    write_manifest(out, payload)
    print(f"wrote {out}")
    print(
        f"available_pdfs={payload['n_available_pdfs']} "
        f"matched={payload['n_matched']} "
        f"selected={payload['n_selected']} "
        f"eval={payload['n_eval']} "
        f"types={len(payload['agreement_types'])}"
    )
    print(f"unmatched={len(payload['unmatched'])} ratio_notes={len(payload['ratio_notes'])}")
    for item in payload["unmatched"]:
        print(f"  skip {item['missing']}: {item['rel_path']}")
    for note in payload["ratio_notes"]:
        print(f"  {note}")
    if payload["n_selected"] < args.target:
        print(f"note: only {payload['n_selected']} joinable docs; target was {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
