# Walkthrough video

Deliverable 2. Screen + narration (Loom, OBS, or Zoom local). **5:00 maximum.**

Required split from the brief:

| Clock | Block | What they score |
|---|---|---|
| 0:00-1:00 | Problem + why Track D | Approach choice |
| 1:00-3:00 | Algorithm decisions + what you ruled out | Why, not code |
| 3:00-4:30 | Results + what failed | Honest error analysis |
| 4:30-5:00 | More time | One concrete next fix |

Live queries are evidence inside those blocks. Talk over the spinner. Do not wait in silence. Do not read code.

Governing-law questions run ~75s (two rewrites). Do **not** ask them live. Lead with the ~30s paths.

After the take: unlisted YouTube or Drive. Put the URL on the README line that says "Walkthrough video (5 min)."

---

## Pre-flight (not recorded)

1. `gpu_default` Streamlit already up. One warm ask already finished (nomic + Qdrant live).
2. Do **not** start a second query, eval, or ingest against that same embedded Qdrant.
3. Sidebar profile stays `gpu_default`.
4. Put this unused CUAD PDF on the desktop (not in the 400-doc index):

   `data/CUAD_v1/full_contract_pdf/Part_III/Maintenance/SECURIANFUNDSTRUST_05_01_2012-EX-99.28.H.9-NET INVESTMENT INCOME MAINTENANCE AGREEMENT.PDF`

5. Have `docs/WRITEUP.pdf` page 1 (Figure 1) ready to flash, then jump back to Streamlit.
6. Have the Experiments page ready so L1 / L2 numbers are one click.
7. Paste the four queries below into a notepad. Copy-paste. Do not type.

If a live ask blows past 45s, skip the next live ask and go to Experiments. The clock wins.

---

## Query card

| # | Path | Paste this | Expect | ~Time |
|---|---|---|---|---|
| 1 | Cited, correct | `Does the Cheetahmobileinc Cooperation Agreement include an Exclusivity clause?` | Caption `corpus_technical`, cites, groundedness near 1. Click a cite. Right pane shows the PDF page. | ~30s |
| 2 | Abstain | `Does the Nelnetinc JOINT FILING AGREEMENT include an Exclusivity clause?` | Caption `corpus_technical · abstained`. Text says the KB has no exclusivity clause. No invented "no" from world knowledge. | ~30s |
| 3 | LLM knowledge | `What is an indemnity clause?` | Caption `general`. Disclaimer that this is not from the corpus. No citations. | ~10-20s |
| 4 | Clarify | `Does this agreement include an Exclusivity clause?` | Caption `ambiguous`. Asks which contract. No retrieve. | ~5-10s |
| 5 | Fresh upload | After ingest: `If the Advantus Money Market Fund's net investment income is below zero on any day, what must Advantus Capital Management do?` | Cite the uploaded Securian NII agreement. Gold: waive advisory fee / reimburse so NII is zero (ss 1.1 / 1.4). | ingest + ~30s |

Spare (only if #1 fails): `Does the Abilityinc Services Agreement include an IP Assignment clause?` -- expect Yes + cite.

Do **not** live-ask: Stamps.com governing law, Telkomsel governing law, MeetGroup governing law. Those burn rewrites and can miss the 5:00 cut. Do **not** live-ask jailbreaks.

---

## Script

### 0:00-1:00 -- Problem and approach

**DO:** Streamlit, **Ask the Corpus**. Sidebar shows `gpu_default`. Click **Documents**, scroll the 400-row table, come back.

**SAY:**

> I built grounded Q and A over commercial contracts. A legal or ops team has hundreds of SEC-filed agreements and needs clause-level answers -- exclusivity, termination, fee waivers -- with a page cite. A fluent paragraph with no evidence is a liability.
>
> Corpus is CUAD v1: 510 contracts, 41 lawyer-labelled clause types, gold spans. I index 400 PDFs, stratified over 25 agreement types. The eval set is 40 items over 30 of those documents.
>
> This is a retrieval problem, not a fine-tune. Track B, a 7B LoRA, would memorize clause phrasing and still invent cites. Track C, a classifier, can tag "has a non-compete" but cannot quote the trigger. Track A, a multi-agent chat, adds messages without adding evidence. Track D is the fit: retrieve the span, generate only from it, abstain when it is missing.

**DO (0:50):** Flash WRITEUP Figure 1 for two seconds if you want, then back to Ask. Paste Query 1. Hit enter.

---

### 1:00-3:00 -- Decisions (talk over the graph)

You have two minutes. Use the spinner. Say the rejection, then point at the screen.

**While Query 1 runs, SAY:**

> Chunking is recursive 512 tokens with 64 overlap, page and bboxes on every chunk. I rejected page-as-chunk because recitals mix with the operative clause, and fixed 512 because it splits mid-sentence. A 600-token indemnity straddles two chunks; the overlap plus a type header recovers it.
>
> Embedder is local nomic 768-d. OpenAI 3-small leaves the clause text. bge-small is 384-d and under-resolves long legal sentences.
>
> Store is embedded Qdrant, dense plus BM25, payload on doc id and type. FAISS cannot delete by payload, so freshness is a full rebuild.

**Query 1 lands. DO:** Point at `corpus_technical`. Click the citation. Point at the highlighted page.

**SAY:**

> That is the happy path. The graph retrieved, graded, generated with forced cites, and the verifier kept the quote.

**DO:** Paste Query 2. Hit enter.

**While Query 2 runs, SAY:**

> Fusion is client RRF. Hybrid lifts dense-only hit-at-5 by 0.20. Sparse-only actually wins this extractive eval set. I still ship hybrid because paraphrase questions -- "what if NII goes negative" -- are the production case. I turned the reranker off. bge-m3 is plus 0.03 hit-at-5 and not worth a cross-encoder inside a 90-second graph.
>
> Control is one LangGraph: Adaptive RAG plus Corrective with a rewrite cap of 2, plus Self-RAG. Naive RAG cannot abstain. A multi-agent chat re-encodes the same four LLM calls. Router, grader, verifier are Haiku. Generation is Sonnet.

**Query 2 lands. DO:** Point at `abstained`.

**SAY:**

> Nelnet Joint Filing has no exclusivity clause in the gold. The system refuses instead of inventing a no from world knowledge. That is the point of the verifier: claims must sit in a retrieved chunk, and the quote must fuzzy-match at 90.

**DO:** Paste Query 3. Hit enter.

**Query 3 lands. SAY:**

> "What is an indemnity clause" never hits the index. Route is general, with a disclaimer. World knowledge is allowed. Pretending it came from a contract is not.

**DO:** Paste Query 4. Hit enter.

**Query 4 lands. SAY:**

> Two words of underspecification and it asks which contract. It does not pick a random Maintenance PDF.

**DO (about 2:35):** Sidebar **Upload a contract** (or Documents). Drop the Securian PDF. Wait for chunk / indexed / seconds metrics. Back to Ask. Paste Query 5.

**While Query 5 / ingest runs, SAY:**

> Freshness is content-addressed. The registry key is sha256 plus index sig. A new PDF upserts; it does not re-embed the 400-doc index. Uploads land as type Unknown. Search ORs that into any type filter so the new contract is visible on the next ask. Same live Qdrant. I do not rebuild the client.

**Query 5 lands. DO:** Click the cite. Point at the uploaded filename.

**SAY:**

> That clause was not in the original 400. The answer is waive the advisory fee or reimburse so net investment income is zero. One ingest, same stack.

If Query 5 is still spinning at 3:00, leave it and jump to Experiments. Come back only if it finishes during the results block.

---

### 3:00-4:30 -- Results and what failed

**DO:** Open **Experiments**. Point at the L1 table, then L2.

**SAY:**

> L1 is deterministic span match, n=30. The runner never injects the gold doc id. FilterExtractor has to find it from the question.
>
> Dense-only hit-at-5 is 0.533. Hybrid RRF, the default, is 0.733. Reranker is 0.767. Sparse-only is 0.800 -- that is the L1 winner, and I am not hiding it. This eval set is extractive. Hybrid stays the production default for paraphrase.
>
> L2 is the same questions plus route and abstain items, n=40. Declared headline is RAGAS faithfulness: 0.636. Custom groundedness 0.605. Route accuracy 1.00. Citation validity 1.00. DeepEval faithfulness is 0.966. It saturates and Spearman with RAGAS is 0.01 on the overlap, so I do not swap the headline after seeing it.
>
> Where it fails: 0.636 means about one third of generated claims are not fully supported -- usually a hedge the verifier lets through, "the agreement appears to." Abstention precision is 0.40. We refuse questions the gold can answer when the grader wants two relevant chunks and the first pass finds one. Governing-law items burn two rewrites and about 75 seconds. That is why I did not demo those live. Parties gold is two-letter fragments; span-match recall is optimistic and I report it separately.

**DO (if 10s left):** Click **Feedback** or the star widget on the last answer. One sentence: ratings land in sqlite, sliced by route and config hash.

---

### 4:30-5:00 -- More time

**DO:** Stay on Experiments or flash WRITEUP section 4. Do not open an IDE.

**SAY:**

> The 400-doc index is about 11k chunks. Adding one contract must not re-embed the rest -- that path already works. At a thousand concurrent queries the bottleneck is 6.5 LLM calls times 48 seconds p50 against provider RPM, not Qdrant.
>
> With more time I would close the 48-second review-tool latency first: a score-threshold first grade so obvious hits skip the rewrite loop, and a tighter rewrite cap. Then server-mode Qdrant, embed behind TEI, and a cache on config hash plus question. No OCR today. Scanned exhibits would miss.

Stop talking at 4:58. Do not "and also."

---

## If you are late

| Time left | Cut |
|---|---|
| Behind at 1:50 | Skip Query 4 (clarify). |
| Behind at 2:20 | Skip upload. Say freshness in one sentence over Experiments. |
| Query 1 still running at 1:50 | Cancel the wait in narration: "this path is retrieve-grade-generate-verify" and jump to Experiments. Do not stack another ask. |
| Upload ingest >20s | Do not ask Query 5. Point at the chunk metrics and move. |

---

## What not to say

- Machine switch, time budget, "hiring challenge," "test set," "finalists."
- Line-by-line code, file paths in `src/`, YAML keys.
- "DeepEval is better so that is the headline."
- "Sparse-only is worse" -- it wins L1; say why hybrid still ships.

---

## Recording notes

- Loom / OBS / Zoom local. Camera optional. Mic close. 1080p.
- Cursor large. Do not mouse-wander during numbers.
