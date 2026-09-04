# Project State Tracker

Companion to `docs/IMPLEMENTATION_PLAN.md`. Update this file at the end of every
working session. One line per task; keep history in the changelog at the bottom.

Status legend: `[ ]` pending  `[~]` in progress  `[x]` done  `[-]` dropped / deferred

Last updated: 2026-09-04 (WS0 Mac acceptance green; Windows CUDA check + C0 PUBLISH pending)

---

## Snapshot

| Workstream | Status | Notes |
|------------|--------|-------|
| WS0 Bootstrap + config | [~] | Mac verified: 13 tests, ruff, mypy --strict green. Open: Windows cu128 check, first COMMIT, C0 PUBLISH |
| WS1 Data manifest + eval set | [ ] | CUAD downloaded locally; PDF/TXT parity verified |
| WS2 Ingestion | [ ] | |
| WS3 Retrieval + L1 eval | [ ] | |
| WS4 LLM + agentic graph | [ ] | |
| WS5 Generation eval (RAGAS + DeepEval) + MLflow + ablations | [ ] | |
| WS6 Feedback DB | [ ] | |
| WS7 Streamlit (incl. Upload PDF) | [ ] | |
| WS8 FastAPI + Qdrant server + load test | [-] | MAY / write-up next step. Docker not required |
| WS9 Write-up + video | [ ] | |

Current focus: WS0

Publish gates (tick only after personal-machine push + Mac `git fetch`):

| Gate | After | On GitHub | Date |
|------|-------|-----------|------|
| C0 | WS0 | [ ] | |
| C1 | WS1 | [ ] | |
| C2 | WS2 | [ ] | |
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
- Windows GPU check: `uv run python -c "import torch;print(torch.__version__, torch.cuda.is_available())"` must show `+cu128` and `True`
- MLflow UI: `uv run mlflow ui --backend-store-uri file:./mlruns` (mlruns/ gitignored)

---

## WS0: Bootstrap + config system

- [x] uv project, `pyproject.toml`, dependency groups, `uv.lock`, `.python-version`
- [ ] torch pinned per platform via `[tool.uv.sources]` (cu128 on win32, cpu on darwin); verified on Windows
- [x] `configs/base.yaml`, `configs/profiles/dev_cpu.yaml`, `gpu_default.yaml`
- [x] `config/schema.py` (strict `extra=forbid` on every node), `config/loader.py`, `config_hash()`, `index_sig()` (ignores `device`/`batch_size`)
- [x] `core/registry.py`, `core/interfaces.py`, `core/types.py`, `core/device.py`, `core/logging.py`
- [x] `cli.py` skeleton (`ingest`, `query`, `eval`, `serve`, `doctor`, `config-hash`); stubs exit 2; `--profile` defaults from `DOCINTEL_PROFILE`
- [x] `Makefile`, `.env.example`, `.pre-commit-config.yaml`
- [x] unit tests: config merge/env override/hash/index_sig/typo rejection; registry; device (13 passed)
- [x] verification pass 2026-09-04: `uv run pytest tests/unit`, `ruff check`, `ruff format --check`, `mypy --strict` (config + core) all green on Mac
- [~] acceptance: Mac green. Windows `uv sync --group gpu` + `+cu128` CUDA print still open
- [ ] first COMMIT (WS0 files are untracked; author confirmed noreply)
- [ ] **C0 PUBLISH**: bundle -> personal push -> Mac fetch. Block WS1 until ticked.

## WS1: Data manifest + eval set

- [ ] `scripts/select_corpus.py` -> `data_manifest/corpus_manifest.json` (400 docs, 50 eval, stratified)
- [ ] stem normalization and join to `master_clauses.csv`; unmatched report
- [ ] `scripts/build_eval_set.py` -> draft questions; hand-review; add general / out-of-scope items
- [ ] document-disjoint split -> `evals/qa_dev.json` (~40, 30 docs) + `evals/qa_test.json` (~30, 20 docs); sha256 in `evals/README.md`
- [ ] all eval-split docs pass PDF/TXT ratio gate before freeze
- [ ] `evaluation/gold.py` span matcher (+ `<omitted>` handling, doc_id match) with unit tests
- [ ] `docs/DATA.md`, `evals/README.md`
- [ ] acceptance: ~70 QA items across 5 buckets (>= 1 per eval doc); disjointness test green; matcher tests green
- [ ] **C1 PUBLISH**: bundle -> personal push -> Mac fetch. Block WS3 L1 until ticked.

## WS2: Ingestion

- [ ] pymupdf loader with page/bbox, header/footer stripping, agreement_type from path
- [ ] txt loader + extraction validation ratio report
- [ ] chunkers MUST: recursive. SHOULD: one of fixed_token / section_aware. MAY: the rest
- [ ] embedders MUST: st_dense (nomic default; bge-small for dev_cpu), openai_embedder (text-embedding-3-small via OPENAI_API_KEY), fastembed_bm25. MAY: bge_m3
- [ ] qdrant indexer: collection named by `index_sig`, fingerprint check, named vectors, payload indexes, uuid5(doc, hash, idx, sig) ids, `delete_by_doc(except_hash)`
- [ ] document registry with `status` + prepare-then-swap order + `--only-changed` / `--resume`
- [ ] `IngestionPipeline` + `ingestion_report.json` (docs, chars, tokens, chunks, bytes, timings, failures)
- [ ] CLI `docintel ingest`, `docintel doctor` (preflight)
- [ ] tests: chunker invariants; embedded-qdrant integration; fault tests (dim mismatch, mid-embed failure keeps old points, second opener)
- [ ] acceptance: `doctor` green on Windows; default-profile ingest (nomic or openai + BM25); incremental re-run skips unchanged; interrupted ingest leaves old version queryable
- [ ] **C2 PUBLISH**: bundle -> personal push -> Mac fetch. No raw ingest reports.

## WS3: Retrieval + L1 eval

- [ ] retrievers: dense, sparse_bm25_inproc, qdrant_hybrid, client_hybrid
- [ ] fusion: rrf, dbsf, weighted
- [ ] rerankers: none (default). MAY: mxbai-xsmall. Do not default to bge-m3 reranker
- [ ] query transforms: filter_extractor, multi_query, hyde
- [ ] `RetrievalPipeline` with per-stage provenance
- [ ] `retrieval_metrics.py` (P@k, R@k, hit@k, MRR, nDCG; per bucket / type)
- [ ] `experiment.py`, `scripts/run_retrieval_eval.py --split dev|test` (per-question checkpoint; test gated by `evals/finalists.txt`), `scripts/make_results_table.py`
- [ ] profiles: exp_dense_only, exp_sparse_only, exp_hybrid_rrf (MUST). DBSF / rerank profiles MAY
- [ ] acceptance: dev L1 table rows 1-3 (dense vs sparse vs hybrid RRF)
- [ ] **C3 PUBLISH**: bundle -> personal push -> Mac fetch. `results/README.md` only; no `per_question.jsonl`.

## WS4: LLM + agentic graph

- [ ] pin exact chat model ids (router/grader/verifier, generation, judge) in `base.yaml`
- [ ] `llm/factory.py` provider-agnostic, role-based; explicit provider from config; fail-fast on missing key; retry policy; `query_deadline_s`
- [ ] `llm/structured.py` + shared pydantic schemas
- [ ] versioned prompts
- [ ] `agent/state.py`, nodes, edges, `graph.py`, `ChunkCache`
- [ ] graders: llm_batch, llm_per_chunk, score_threshold
- [ ] verifiers: llm_claims, nli_cross_encoder, lexical_overlap; citation validation
- [ ] abstain / general / clarify / refuse nodes
- [ ] `QueryService.ask()` + query logging + JSONL trace sink + `mlflow.langchain.autolog()` bootstrap
- [ ] tests: fake-LLM graph; factory provider selection; malformed structured output; API timeout; adversarial-chunk fixture
- [ ] acceptance: cited answer, abstention, general answer via CLI; provider switch verified for: [ ] anthropic [ ] openai [ ] google_genai (unverified providers marked as such)
- [ ] saved traces: [ ] success path [ ] abstain path
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
| 2 | Pinned Anthropic model ids per role | OPEN (haiku class for router/grader/verifier; sonnet class for generation/judge) | |
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
