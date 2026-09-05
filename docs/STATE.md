# Project State Tracker

Companion to `docs/IMPLEMENTATION_PLAN.md`. Update this file at the end of every
working session. One line per task; keep history in the changelog at the bottom.

Status legend: `[ ]` pending  `[~]` in progress  `[x]` done  `[-]` dropped / deferred

Last updated: 2026-09-05 (WS4 code + review on Mac; live CLI still Windows; C4 not published)

---

## Snapshot

| Workstream | Status | Notes |
|------------|--------|-------|
| WS0 Bootstrap + config | [x] | Mac tests/ruff/mypy green. Windows `2.11.0+cu128 True NVIDIA GeForce RTX 5060 Laptop GPU`. C0 = `2acf9af` |
| WS1 Data manifest + eval set | [x] | 510 walked, 400/50 manifest, 25 types, group-balanced qa_dev 40 / qa_test 30. C1 = `a5b134b` |
| WS2 Ingestion | [x] | C2 = `5be2905` (Windows cleanup + embedding API). `gpu_default` nomic 400-doc ingest finished |
| WS3 Retrieval + L1 eval | [x] | Dev L1 on origin (`523d1cd`). Sparse leads; bge-m3 helps hybrid; bge-base loses to RRF. C3 not ticked |
| WS4 LLM + agentic graph | [~] | Factory + LangGraph + QueryService + fake-LLM tests on Mac. Live CLI / provider keys unverified |
| WS5 Generation eval (RAGAS + DeepEval) + MLflow + ablations | [ ] | |
| WS6 Feedback DB | [ ] | |
| WS7 Streamlit (incl. Upload PDF) | [ ] | |
| WS8 FastAPI + Qdrant server + load test | [-] | MAY / write-up next step. Docker not required |
| WS9 Write-up + video | [ ] | |

Current focus: WS4 live CLI smoke (`docintel query` with `ANTHROPIC_API_KEY`) then C4 traces. Do not start L2 / WS5 until cited + abstain + general work against a real index

Publish gates (tick only after personal-machine push + Mac `git fetch`):

| Gate | After | On GitHub | Date |
|------|-------|-----------|------|
| C0 | WS0 | [x] | 2026-09-04 `2acf9af` |
| C1 | WS1 | [x] | 2026-09-04 `a5b134b` |
| C2 | WS2 | [x] | 2026-09-05 `5be2905` |
| C3 | WS3 | [ ] | |
| C4 | WS4 | [ ] | |
| C5 | WS5 | [ ] | |
| C6 | WS6 | [ ] | skip if deferred |
| C7 | WS7 | [ ] | skip if deferred |
| C8 | WS8 | [-] | MAY |
| C9 | WS9 | [ ] | submission snapshot |

---

## Decisions log (durable)

| Date | Decision | Reason |
|------|----------|--------|
| 2026-09-03 | Track D + S2, corpus CUAD v1 | gold clause spans give deterministic P@k / R@k; contract types align with procurement / supplier docs |
| 2026-09-03 | Index PDFs, not TXT | page + bbox needed for citation highlights; TXT kept as extraction QA oracle |
| 2026-09-03 | Gold matching by normalized text, not `answer_start` | offsets refer to TXT-derived context, do not align to PDF text |
| 2026-09-03 | Qdrant (embedded dev, server later) | native dense+sparse hybrid, RRF/DBSF, idempotent upsert, payload filters |
| 2026-09-03 | bge-m3 + bge-reranker-v2-m3 default; Qwen3-Embedding-0.6B, mxbai, Qwen3-Reranker as ablations | superseded 2026-09-04 |
| 2026-09-04 | Default: nomic-v1.5 or OpenAI text-embedding-3-small + fastembed BM25 + reranker none | challenge wants a workable system and known limits, not max accuracy / slow inference |
| 2026-09-04 | OpenAI embedder is first-class (`dense_embedder.name: openai` + `OPENAI_API_KEY` in `.env`) | owner may obtain an OpenAI key; same .env convention as LLM keys |
| 2026-09-04 | WS8 + Docker deferred to write-up next steps | not required; Qdrant embedded covers the demo |
| 2026-09-05 | Chat ids pinned: haiku-4-5 router/grader/verifier; sonnet-4-6 generation/judge | `init_chat_model("provider:model")`; owner had these in `base.yaml` |
| 2026-09-03 | LLM via `init_chat_model("provider:model")`, role-based models | provider switch by config + env key only |
| 2026-09-03 | RAGAS 0.4 and DeepEval both planned behind `BaseGenerationEvaluator`; custom deterministic metrics for L1 | cheap L1 for every ablation; LLM judge (Anthropic) only for finalists |
| 2026-09-04 | RAGAS faithfulness is the declared headline; DeepEval is a SHOULD cross-check | avoid choosing the metric after seeing results |
| 2026-09-04 | Eval split: document-disjoint `qa_dev` (~40) / `qa_test` (~30); test run once per finalist | rubric "no test leakage" |
| 2026-09-04 | Collection named by `index_sig`; prepare-then-swap incremental ingest; registry `status` + `--resume` | plan review: alias collisions and delete-before-embed data loss |
| 2026-09-04 | 72h budget confirmed; agentic default path (classify/grade/generate/verify + rewrite) kept | owner decision |
| 2026-09-03 | Corpus 400 docs / 50 held out (stratified, Avathon-weighted eval sampling) | owner decision; enough decoys for realistic retrieval |
| 2026-09-03 | MLflow on: experiment tracking + LangGraph tracing; JSONL app trace; Phoenix optional; no LangSmith / hosted Langfuse | owner requires open-source, local tracing |
| 2026-09-03 | Streamlit Upload PDF demo in scope; WS8 in scope, minimal | superseded 2026-09-04: WS8 is MAY |
| 2026-09-03 | Code on macOS, all GPU runs on Windows (Core Ultra 7, RTX 5060 8 GB); torch>=2.7 cu128 pinned via uv sources | Blackwell sm_120 needs cu128 wheels |
| 2026-09-03 | Repo pushes only from personal machine | org pre-push hook blocks personal GitHub destinations from work laptop |

---

## Repo / environment notes

- Remote: `https://github.com/abhikdebnath24/document-intelligence.git` (private)
- Local git author in this repo: `49766667+abhikdebnath24@users.noreply.github.com` (set; do not commit with work email)
- Push flow (never `git push` from this laptop):
  1. COMMIT on Mac after each task (author must be the noreply address)
  2. PUBLISH at C0-C9 and at session end: `git bundle create ../document-intelligence.bundle --all`
  3. Copy the bundle to the personal machine (not `data/`, not `.env`)
  4. Personal: fetch the bundle, `git push -u origin HEAD`
  5. Mac: `git fetch origin` and fast-forward
- Plan 11.0 has the gate table. Tick the Snapshot publish-gate row when GitHub has the commits.
- Data: `data/CUAD_v1/` (gitignored, 168 MB unzipped)
- Hardware: macOS for writing code + `dev_cpu` smoke runs; Windows (Core Ultra 7, RTX 5060 8 GB) for ingest, ablations, evals, demo
- Windows GPU check: passed 2026-09-04. `2.11.0+cu128 True NVIDIA GeForce RTX 5060 Laptop GPU`
- MLflow UI: `uv run mlflow ui --backend-store-uri file:./mlruns` (mlruns/ gitignored)

---

## WS0: Bootstrap + config system

- [x] uv project, `pyproject.toml`, dependency groups, `uv.lock`, `.python-version`
- [x] torch pinned per platform via `[tool.uv.sources]` (cu128 on win32, cpu on darwin); verified on Windows (`2.11.0+cu128`)
- [x] `configs/base.yaml`, `configs/profiles/dev_cpu.yaml`, `gpu_default.yaml`
- [x] `config/schema.py` (strict `extra=forbid` on every node), `config/loader.py`, `config_hash()`, `index_sig()` (ignores `device`/`batch_size`)
- [x] `core/registry.py`, `core/interfaces.py`, `core/types.py`, `core/device.py`, `core/logging.py`
- [x] `cli.py` skeleton (`ingest`, `query`, `eval`, `serve`, `doctor`, `config-hash`); stubs exit 2; `--profile` defaults from `DOCINTEL_PROFILE`
- [x] `Makefile`, `.env.example`, `.pre-commit-config.yaml`
- [x] unit tests: config merge/env override/hash/index_sig/typo rejection; registry; device (13 passed)
- [x] verification pass 2026-09-04: `uv run pytest tests/unit`, `ruff check`, `ruff format --check`, `mypy --strict` (config + core) all green on Mac
- [x] acceptance: Mac green. Windows `uv sync --group gpu` + `2.11.0+cu128 True NVIDIA GeForce RTX 5060 Laptop GPU`
- [x] first COMMIT `2acf9af` (noreply author + committer)
- [x] **C0 PUBLISH**: bundle -> personal push -> Mac `git fetch`. `origin/main` == `2acf9af`

## WS1: Data manifest + eval set

- [x] `scripts/select_corpus.py` -> `data_manifest/corpus_manifest.json` (400 docs, 50 eval, 25 types, seed 42)
- [x] stem normalization + alnum key join to `master_clauses.csv`; sha256 dedupe; 4 skipped (3 no CSV row, 1 duplicate) reported in the manifest
- [x] eval docs restricted to single-member stem families (no `_part1/_part2`, `agreement1/3` siblings as decoys with the same name)
- [x] `scripts/build_eval_set.py` templates + 4 general/out-of-scope items (wording editable in JSON); clause cells parsed with `ast.literal_eval`
- [x] document-disjoint, group-balanced `evals/qa_dev.json` (40 / 30 docs: 14 core, 10 IP, 6 other) + `evals/qa_test.json` (30 / 20 docs: 10 / 6 / 4); sha256 in `evals/README.md`
- [x] all 50 eval docs passed PDF/TXT ratio gate (no replacements)
- [x] `evaluation/gold.py` SpanMatcher (`<omitted>`, hyphenation, doc_id); unit tests
- [x] `docs/DATA.md`, `evals/README.md`
- [x] acceptance: 70 QA across 5 buckets; >= 1 question per eval doc; disjointness + matcher tests green
- [x] **C1 PUBLISH**: `origin/main` = `a5b134b` (noreply). WS2 unblocked.

## WS2: Ingestion

- [x] pymupdf loader with page/bbox, header/footer stripping, agreement_type from path
- [x] txt loader + extraction validation ratio report
- [x] chunkers MUST: recursive. SHOULD: fixed_token (`exp_chunk_fixed`). MAY skipped
- [x] embedders MUST: `st_dense` / `nomic_v15` (sentence-transformers, no `trust_remote_code`), `openai` (`embeddings.create` + `dimensions`, openai 2.54), `fastembed_bm25` (`Qdrant/bm25` + `Modifier.IDF`). `hash` / `hash_sparse` for tests. MAY bge_m3 skipped
- [x] qdrant indexer (client 1.19): `cuad__{index_sig[:12]}`, fingerprint point, named `dense`+`sparse`, payload indexes, uuid5 ids, `count_by_doc_hash`, `delete_by_doc(except_hash)`, `query_points`
- [x] document registry (sqlite3) with `status` + prepare-then-swap + `--only-changed` (default) / `--full`; failed rows auto-retried
- [x] `IngestionPipeline` + `.cache/ingestion_report.json`
- [x] CLI `docintel ingest --path/--only-changed|--full/--report`, `docintel doctor`
- [x] tests: chunker invariants; header strip reaches blocks; 3-PDF embedded Qdrant; dim mismatch; mid-embed keeps old points then retries; second opener `QdrantInUseError`; manifest jobs. 47 passed; `mypy --strict` covers `docintel.ingestion`
- [x] review pass: header strip now at block level (was dead on the PDF path); ratio gate uses pre-strip text; spans joined without spaces; page-number regex capped at 3 digits; `--only-changed` gate fixed and failed rows retried; nomic `trust_remote_code` fallback. Verified on 40 real CUAD PDFs: 0 ratio failures, ~48k header chars stripped, all chunks have bboxes
- [x] acceptance: Windows `doctor --profile gpu_default` green. 400-doc nomic ingest finished
- [x] **C2 PUBLISH**: `origin/main` = `5be2905`. WS3 unblocked.

## WS3: Retrieval + L1 eval

- [x] retrievers: dense, sparse (Qdrant named BM25), qdrant_hybrid (`RrfQuery` / `FusionQuery.DBSF`), client_hybrid. `sparse_bm25_inproc` implemented, not the default sparse path
- [x] fusion: rrf (k=60), dbsf (Qdrant 3-sigma), weighted(alpha)
- [x] rerankers: none (default). MAY: `cross_encoder` (`exp_hybrid_rerank_bge_base` = `BAAI/bge-reranker-base`, `exp_hybrid_rerank_bge` = `BAAI/bge-reranker-v2-m3`). Fetch 20, rerank to 10. Same index_sig as nomic ingest.
- [x] query transforms: filter_extractor (company-segment catalog match -> `doc_id` or `doc_id IN` candidates; type from catalog; CUAD clause vocab stop-listed; no gold doc_id). On the 66 scored eval questions: 53 single doc, 12 candidate sets, 0 wrong filters (tested). multi_query / hyde are identity until WS4 LLM
- [x] `RetrievalPipeline` with per-stage provenance
- [x] `retrieval_metrics.py` (P@k, R@k any/all, hit@k, MRR, nDCG; per bucket / type). Abstention / general excluded
- [x] `experiment.py`, `scripts/run_retrieval_eval.py --split dev|test` (per-question checkpoint; test gated by `evals/finalists.txt`), `scripts/make_results_table.py`
- [x] profiles: exp_dense_only, exp_sparse_only, exp_hybrid_rrf, exp_hybrid_rerank_bge_base, exp_hybrid_rerank_bge (same `index_sig` as nomic ingest). DBSF still MAY
- [x] acceptance: dev L1 table rows 1-3 + bge-base + bge-m3 on Windows; same `index_sig` `be217ccb7628`
- [ ] **C3 PUBLISH**: table is on `origin/main`; tick after treating that push as the C3 gate. `results/README.md` only; no `per_question.jsonl`.

## WS4: LLM + agentic graph

- [x] pin exact chat model ids in `base.yaml` (already present): router/grader/verifier `anthropic:claude-haiku-4-5`; generation/judge `anthropic:claude-sonnet-4-6`
- [x] `llm/factory.py`: `init_chat_model("provider:model")`; key from config only; fail-fast on missing env; retry 429/5xx/timeout with jitter; no retry on 401/403; `query_deadline_s` on nodes
- [x] `llm/structured.py` + schemas; JSON fence / trailing-comma repair
- [x] versioned prompts under `llm/prompts/` (`prompt_version` in trace)
- [x] `agent/state.py`, `nodes.py`, `edges.py`, `graph.py` (LangGraph 1.2 `StateGraph` + `START`/`END`), `ChunkCache`
- [x] graders: `llm_batch` (default), `llm_per_chunk`, `score_threshold`
- [x] verifiers: `llm_claims` (default), `nli_cross_encoder`, `lexical_overlap`; citation validation always on
- [x] abstain / general / clarify / refuse nodes
- [x] `QueryService.ask()` + in-memory `QueryLog` + JSONL sink + optional `mlflow.langchain.autolog()` (skipped if mlflow missing)
- [x] tests: fake-LLM graph (cited / abstain / general / refuse / clarify / rewrite cap / first-pass keep); factory missing key + retry + repair; adversarial-chunk fixture; 78 unit+graph tests green
- [ ] acceptance: cited answer, abstention, general answer via live CLI; provider switch verified for: [ ] anthropic [ ] openai [ ] google_genai (unverified)
- [x] saved traces: `evals/curated_traces/success.jsonl` [x] `evals/curated_traces/abstain.jsonl` (scripted; replace with live traces before C4)
- [ ] **C4 PUBLISH**: bundle -> personal push -> Mac fetch. Two curated traces only.

## WS5: Generation eval (RAGAS + DeepEval) + MLflow + ablations

- [ ] `frameworks/base.py` (`BaseGenerationEvaluator`, `EvalSample`, `EvalResult`)
- [ ] `ragas_adapter.py` (collections API, judge via llm_factory)
- [ ] `deepeval_adapter.py` (`DeepEvalBaseLLM` wrapper, 5 RAG metrics + G-Eval rubric)
- [ ] `custom_metrics.py`: route accuracy, abstention P/R, citation validity, latency, tokens
- [ ] `tracking.py`: MLflow run per experiment (params, metrics, artifacts, tags)
- [ ] ablation ladder on `qa_dev`: rows 1-3 + 7 (MUST), rows 4-5 (SHOULD), rows 6, 8 (MAY)
- [ ] `evals/finalists.txt` locked; finalists run once on `qa_test` (L1 + RAGAS)
- [ ] `results/framework_agreement.md` (RAGAS vs DeepEval; SHOULD)
- [ ] `scripts/run_demo_queries.py` -> `results/demo_queries.md` (>= 5 queries)
- [ ] error analysis: top-20 misses with reason tags
- [ ] acceptance: `results/README.md` complete with config_hash, qa_sha, index_sig, MLflow run ids, per-stage p50/p95, llm_calls_per_query, tokens; `mlflow ui` shows runs and traces
- [x] headline eval framework: RAGAS faithfulness (declared up front; DeepEval is a cross-check)
- [ ] **C5 PUBLISH**: bundle -> personal push -> Mac fetch. Sanitized results + `demo_queries.md`; no `mlruns/`.

## WS6: Feedback DB

- [ ] SQLAlchemy models: documents, query_logs, feedback
- [ ] repository ABC + SQLAlchemy impl; `FeedbackService`
- [ ] `scripts/analyze_feedback.py`
- [ ] tests (in-memory SQLite)
- [ ] **C6 PUBLISH** (skip if WS6 deferred)

## WS7: Streamlit

- [ ] client abstraction (inprocess / http)
- [ ] chat page: answer, route badge, citations, trace expander
- [ ] PDF viewer with bbox highlights
- [ ] feedback widget
- [ ] documents page with Upload PDF (uuid filename, magic bytes, size/page caps, external-API notice) -> incremental ingest -> immediately queryable
- [ ] Streamlit bound to 127.0.0.1
- [ ] experiments page (profile picker, results comparison, link to mlflow ui)
- [ ] feedback analytics page
- [ ] acceptance: end-to-end demo path works
- [ ] **C7 PUBLISH** (skip if WS7 deferred)

## WS8: FastAPI + Qdrant server + load test (MAY / deferred)

- [-] FastAPI `/query` `/feedback` `/ingest` `/health` (async)
- [-] docker-compose: qdrant server (not needed for embedded Qdrant)
- [-] `vectorstore.params.mode: server` profile
- [-] `frontend.backend: http` path
- [-] `scripts/load_test.py` 10/50/100 concurrency
- Write-up covers the scale path even if this code is not written
- [-] **C8 PUBLISH**: skipped (MAY)

## WS9: Write-up + video

- [ ] README (setup, reproduce, results, external-API disclosure, video link)
- [ ] `write-up/writeup.md` -> PDF (1-2 pages, 11pt, name + "Track D" at top; test-split numbers; capacity worksheet)
- [ ] `docs/VIDEO_SCRIPT.md`
- [ ] `docintel verify-demo` green; video recorded and linked
- [ ] `uv export --frozen --no-dev -o requirements.txt` committed with `.python-version`
- [ ] reviewer GitHub handle added to private repo; access confirmed
- [ ] `AI_Challenge.pdf` confirmed absent from repo and video
- [ ] **C9 PUBLISH**: submission snapshot (README, write-up, `requirements.txt`, video link). No C8.

---

## Open questions (mirror of plan section 14)

| # | Question | Answer | Date |
|---|----------|--------|------|
| 1 | Corpus size / types | 400 indexed, 50 held out, stratified | 2026-09-03 |
| 2 | Pinned Anthropic model ids per role | `claude-haiku-4-5` router/grader/verifier; `claude-sonnet-4-6` generation/judge | 2026-09-05 |
| 3 | Eval judge provider / frameworks | Same Anthropic key; RAGAS headline, DeepEval cross-check | 2026-09-04 |
| 4 | Tracing sink | Open source only: JSONL + MLflow autolog; Phoenix optional | 2026-09-03 |
| 5 | MLflow on/off | On | 2026-09-03 |
| 6 | Streamlit upload demo | Yes (SHOULD) | 2026-09-03 |
| 7 | WS8 in scope? | No. MAY / write-up next steps. Docker not required | 2026-09-04 |
| 8 | Package/CLI name `docintel` | Keep (Python package + CLI command) | 2026-09-03 |
| 9 | Where full runs execute | Windows RTX 5060; Mac dev only | 2026-09-03 |
| 10 | jina-reranker-v3.5 ablation despite CC-BY-NC | Keep as optional ablation | 2026-09-03 |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-09-03 | Plan and state tracker created. Repo recreated as `document-intelligence`; `data/` gitignored; CUAD v1 downloaded (168 MB). PDF vs TXT parity verified on 12 samples. |
| 2026-09-03 | Owner decisions applied: 400/50 corpus, RAGAS + DeepEval, MLflow tracking + tracing, no paid tracing, Upload PDF demo, WS8 in scope, Windows GPU for all runs, cu128 torch pin, jina reranker as option. |
| 2026-09-04 | Scope cut: WS8 + Docker deferred; default models nomic/openai-3-small + BM25 + no reranker; OpenAI embedder via .env; mermaid gantt replaced with flowchart. |
| 2026-09-04 | Plan review applied: dev/test eval split, `index_sig` collections, prepare-then-swap ingest, `doctor` / `verify-demo` preflights, explicit provider selection, judge role in base config, nomic prefixes + revision, capacity worksheet, artifact allowlist, upload validation, submission gates (requirements.txt, reviewer access, name + track), runtime artifacts gitignored. 48h point rejected (72h confirmed); agentic default path kept. |
| 2026-09-04 | C0-C9 COMMIT/PUBLISH gates added. COMMIT after each task on Mac; PUBLISH = bundle + personal push + Mac fetch. |
| 2026-09-04 | WS0 implemented on Mac: uv project, config loader + hash/index_sig, registry, ABCs, CLI stubs, 11 unit tests. Windows cu128 verify and C0 PUBLISH still open. |
| 2026-09-04 | WS0 verification: fixed 4 mypy strict errors, `typer.Exit(str)` misuse, `index_sig` now excludes runtime knobs, `NamedStrategy` extra=forbid, CLI profile default from `DOCINTEL_PROFILE`. 13 tests green. |
| 2026-09-04 | C0 closed. Windows CUDA `2.11.0+cu128 True RTX 5060 Laptop GPU`. `origin/main` = `2acf9af` (noreply). WS1 unblocked. |
| 2026-09-04 | WS1: 400/50 stratified manifest (25 types, 3 unmatched), ratio gate green on all 50 eval docs, qa_dev 40 / qa_test 30 document-disjoint, SpanMatcher tests. C1 not published. |
| 2026-09-04 | WS1 review: 510 PDFs walked by suffix (311 `.PDF`); byte-identical duplicate dropped; eval docs must be sole member of their stem family; hyphen join limited to line breaks; dead helpers removed; `txt_name` added to manifest. 30 tests green. |
| 2026-09-04 | WS1: 400 top-up/trim now from largest types; 30/20 split is seeded and group-balanced (14/10/6 and 10/6/4). 32 tests green. |
| 2026-09-04 | WS1 fix: CSV clause cells are Python reprs; `gold_spans` were stored as `"['...']"` strings. Parse with `ast.literal_eval`; eval JSONs regenerated. Plan 3.1/3.3/3.4/WS1 rewritten to match the build. 33 tests green. WS1 COMMIT on Mac. |
| 2026-09-04 | C1 closed. Personal push landed. Mac `git fetch` + ff: `origin/main` = `a5b134b` (noreply). WS2 unblocked. |
| 2026-09-04 | WS2 on Mac: pymupdf+txt loaders, recursive+fixed_token, st/nomic/openai/fastembed BM25 (IDF), Qdrant 1.19 named vectors + prepare-then-swap registry, doctor. 44 tests. C2 blocked on Windows doctor + full ingest. |
| 2026-09-04 | WS2 review: fixed dead header strip (blocks), ratio on pre-strip text, span join, page-number regex, `--only-changed` no-op + silent failed skips (`--resume` dropped), doctor hash false-fail, nomic remote-code fallback. 47 tests; mypy strict on ingestion. |
| 2026-09-05 | C2 closed. Personal push landed. Mac `git fetch` + ff: `origin/main` = `f7908d4` (noreply). Windows doctor green; 400-doc ingest still to finish after reboot. |
| 2026-09-05 | Windows ingest cleanup + embedding API warning: `origin/main` = `5be2905`. 400-doc nomic ingest finished. |
| 2026-09-05 | WS3 on Mac: Qdrant 1.19 hybrid (`RrfQuery` / DBSF), client RRF/DBSF/weighted, filter_extractor, L1 metrics + checkpointed runner, three exp profiles. LangGraph deferred to WS4. L1 table not yet run. |
| 2026-09-05 | WS3 review: chunk relevance = any whole gold span (was all; zeroed 31/52 multi-span questions); doc hint matches the company segment only, clause words stop-listed, ambiguous company -> `MatchAny` doc set instead of a title-derived type guess (title vs folder type disagreed 1/18); eval fails loudly on a missing collection; `--scoped` gets its own run dir; split validated; scroll `Record` has no score; fusion provenance is a union. 63 tests. |
| 2026-09-05 | WS3 L1 table on origin (`523d1cd`): sparse best rank; hybrid+bge-m3 best r@10; bge-base below RRF (slot collapse). |
| 2026-09-05 | WS4 started on Mac: LangGraph 1.2 StateGraph, `init_chat_model`, QueryService, fake-LLM tests. Live CLI + provider keys still open. 76 tests; mypy on llm/agent/service/core green. |
| 2026-09-05 | WS4 review: router `doc_hint`/`agreement_type` no longer become Qdrant payload filters (unknown key = zero hits = forced abstain); rewrite keeps first-pass candidates; 1 relevant chunk after the rewrite cap answers instead of abstaining; `structured()` falls back to raw+repair only on parse errors (API errors retry, not double-call); mlflow autolog imports `mlflow.langchain`; CLI catches `MissingSecretError` at container build. 78 tests. |
