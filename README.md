# SUSC Knowledge Assistant (RAG)

A question-answering assistant over **SUSC's own documents** (constitution, event
SOPs, EMR process, orientation/logistics). It answers **only from the indexed
sources, with citations**, and **refuses politely** when the answer isn't in the
documents. The part that makes it more than a chatbot is the **evaluation
harness** (retrieval + answer metrics) and an **ablation study** showing what
actually improved retrieval.

> **Live demo:** _add your Vercel URL here after deploying_ — e.g. `https://susc-rag.vercel.app`
>
> ⚠️ The documents in `data/public/` are **SAMPLE placeholders** so the project
> runs end-to-end out of the box. Replace them with real, cleared SUSC documents
> and re-run the pipeline (`python -m src.ingest.pipeline --full`).

---

## 1. What it does

Ask a question → the system retrieves the most relevant chunks of SUSC documents
(semantic search, with keyword/hybrid options evaluated) → an LLM answers using only those chunks,
citing each as `[1]`, `[2]` → if nothing relevant is found, it refuses instead
of guessing.

`POST /api/ask` returns the answer, the cited sources (file + section), whether
it refused, and timing per stage.

## 2. Architecture

Two clearly separated worlds. This split is deliberate: production RAG systems
almost never re-embed documents inside the request path.

```
OFFLINE — your laptop (full tooling, no size limits)
  SUSC docs → parse → clean → chunk → embed (sentence-transformers)
                                           │
            ┌──────────────────────────────┴───────────────────────────┐
            │ EXPORT (committed, small)                                  │
            │   index/chunks.json     chunk text + metadata              │
            │   index/vectors.npy     precomputed embeddings             │
            │   index/bm25_corpus.json tokenized corpus for rank_bm25    │
            └────────────────────────────────────────────────────────────┘
      Also local-only: ChromaDB / cross-encoder reranker for the ablation study.

                              git push ↓

ONLINE — Vercel (one Python serverless function, slim deps, no PyTorch)
  question → api/index.py
    1. load index/ once per cold start (in-memory, read-only)
    2. embed the question with fastembed (ONNX, same model as offline)
    3. retrieve top-5 (dense NumPy cosine by default; hybrid RRF / BM25 selectable)
    4. build prompt (rules + numbered sources + question)
    5. call Groq (free tier) for the answer
    6. return {answer, sources[], refused, timings_ms}
  static frontend: public/index.html on the same Vercel project
```

**Shared code is the key.** `src/retrieval/bm25.py`, `hybrid.py`,
`generation/prompts.py`, `generation/llm.py`, and `retrieval/engine.py` are
written once and imported by **both** the offline eval scripts and the live
`api/index.py`. The local ablation numbers are therefore honestly
representative of what is actually deployed.

## 3. Quickstart (local, < 15 min)

Requires Python 3.11+ and a free [Groq API key](https://console.groq.com/keys)
(email only, no card).

```bash
# 1. Environment
python -m venv .venv
.venv/Scripts/activate            # Windows (PowerShell/Git Bash)
# source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt   # heavy offline stack (one-time)

# 2. Secrets
cp .env.example .env              # then paste your GROQ_API_KEY into .env

# 3. Build the index from data/public/
python -m src.ingest.pipeline --full

# 4. Sanity-check retrieval (no LLM, no key needed)
python scripts/query_cli.py "What does EMR stand for?"

# 5. Ask end-to-end (uses the LLM)
python scripts/ask_cli.py "Who chairs the EMR shortlisting process?"

# 6. Run the API locally
.venv/Scripts/uvicorn api.index:app --reload --port 8000
# open http://localhost:8000/  (served from public/index.html via the frontend,
# or hit the API directly: curl -X POST localhost:8000/api/ask -H "content-type: application/json" -d "{\"question\":\"When is orientation?\"}")
```

> To serve the static frontend together with the API locally, open
> `public/index.html` in a browser and point its `fetch` at `localhost:8000`,
> or just use the deployed Vercel URL where routing is configured in `vercel.json`.

### Local LLM fallback (no Groq, offline dev)

Install [Ollama](https://ollama.com), `ollama pull gemma3:4b`, then in `.env`:

```
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=gemma3:4b
GROQ_API_KEY=ollama
```

The same `chat()` function talks to both — only config changes. (Ollama is
dev-only; it never runs on Vercel.)

## 4. Deployment (Vercel Hobby, free, no card)

1. Push this repo to GitHub (public).
2. Import it at [vercel.com](https://vercel.com) (GitHub sign-in).
3. Set environment variables in the Vercel dashboard: `GROQ_API_KEY`,
   `LLM_MODEL`, and optionally `RETRIEVAL_MODE`.
4. Deploy. `vercel.json` routes `/api/*` to the Python function and serves
   `public/` statically. Only `api/requirements.txt` (slim) affects the function.

**Size limit:** the function must stay under 250 MB unzipped. Run the check
**early** (see [`scripts/size_check.md`](scripts/size_check.md) and
`python scripts/estimate_api_size.py`). If `fastembed` doesn't fit, set
`RETRIEVAL_MODE=bm25` in Vercel — the engine auto-degrades to BM25-only and the
full dense/hybrid pipeline stays in the local ablation study.

## 5. Design decisions

| Decision | Why |
|---|---|
| Precompute index offline, bundle small files | Vercel functions are stateless, read-only at request time, size-capped. A few-hundred-chunk corpus needs no DB — NumPy cosine + BM25 in memory is faster and simpler. |
| sentence-transformers offline, fastembed online | Same model (`bge-small-en-v1.5`), one vector space. fastembed is ONNX (no PyTorch), so the live function stays small. |
| **Dense** retrieval in production (chosen by ablation), with BM25-only auto-fallback | Hybrid and the reranker are fully implemented, but the ablation on the real corpus showed dense alone wins (see Evaluation). Dense is deployed; if `fastembed` fails or doesn't fit, the engine degrades gracefully to BM25-only. Mode is one config line / env var. |
| Own recursive chunker (~120 lines) | Chunking is the biggest lever on quality; worth owning and explaining rather than hiding in a framework. |
| Refuse below a confidence threshold, before the LLM | Cheap, deterministic hallucination guard. Threshold tuned on the eval set (Week 5). |

## 6. Evaluation

- **Dataset:** `src/eval/dataset.jsonl` — hand-written questions with gold
  answers and gold sources, typed (factual, keyword/acronym, paraphrase,
  multi-chunk, unanswerable). _Scaffolded with example questions against the
  sample corpus; expand to ≥ 40 (incl. ≥ 8 unanswerable) against your real docs._
- **Retrieval metrics** (`src/eval/retrieval_eval.py`): hit@1/3/5, MRR. No API key needed.
- **Answer metrics** (`src/eval/answer_eval.py`): citation rate, refusal
  accuracy (automatic), plus LLM-as-judge groundedness & correctness (temp 0,
  fixed rubric, responses cached). Grade ~20 yourself and compare agreement.
- **Ablations** (`src/eval/run_ablations.py`): dense vs BM25 vs hybrid × chunk
  sizes × reranker on/off.

```bash
python -m src.eval.retrieval_eval --save
python -m src.eval.answer_eval --no-judge        # layer (a); add key for the judge
python -m src.eval.run_ablations                 # heavy; writes CSV + markdown
```

### Baseline results

Measured on the **real SUSC corpus** (11 files → 56 chunks; 36 answerable + 8
unanswerable questions). Deployed config = **dense** (see ablation below).
Re-run `python -m src.eval.retrieval_eval --save` after any corpus change.

| Mode | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| **dense (deployed)** | 0.667 | **0.861** | **0.917** | **0.773** |
| hybrid | 0.639 | 0.833 | 0.833 | 0.736 |
| BM25 | 0.500 | 0.694 | 0.833 | 0.625 |

> Retrieval-only metrics (no API key). Answer-quality metrics (citation rate,
> refusal accuracy, LLM-judge groundedness/correctness) run via
> `python -m src.eval.answer_eval` once `GROQ_API_KEY` is set.

### Ablation table (local study)

From `src/eval/runs/ablations_*.md` on the real corpus. **Row B is the deployed
configuration.**

| Config | Chunk size | Retrieval | Rerank | hit@3 | MRR | p50 retrieve (ms) |
|---|---|---|---|---|---|---|
| A | 1200 | dense | no | 0.861 | 0.772 | 18.7 |
| **B** | **2000** | **dense** | **no** | **0.861** | **0.773** | **18.1** ← deployed |
| C | 3200 | dense | no | 0.861 | 0.772 | 19.0 |
| D | 2000 | BM25 | no | 0.694 | 0.625 | 0.2 |
| E | 2000 | hybrid | no | 0.833 | 0.732 | 17.9 |
| F | 2000 | hybrid | yes | 0.861 | 0.782 | 1431.6 |

**What this shows (and why dense is deployed):**
- **Dense wins on this corpus.** The SUSC documents are paraphrase-heavy training
  prose, where semantic search excels. BM25's weaker keyword rankings (D, hit@3
  0.69) actually *drag down* the RRF fusion, so plain hybrid (E, 0.83) is worse
  than dense alone (B, 0.86). This is the honest opposite of the textbook "hybrid
  always wins" — and exactly why the project measures instead of assuming.
- **The reranker (F) recovers hybrid back to dense's hit@3 (0.861)** but costs
  **~80× latency** (1432 ms vs 18 ms) and still doesn't beat plain dense. Clear
  conclusion: **not worth deploying.**
- **Chunk size is irrelevant here** (A/B/C identical at 0.861) — sections are
  short and single-topic, so the splitter rarely has to cut mid-fact.
- **BM25 stays as the production fallback** (0.2 ms, zero model) for the case
  where `fastembed` doesn't fit Vercel's size limit — a graceful, documented
  degradation, not the primary path.

> Note: the deployed numbers use `fastembed` (ONNX) for queries; the ablation
> uses `sentence-transformers` (PyTorch) offline. Same model, near-identical
> vectors — the small print is documented rather than hidden.

### Deployment size

`python scripts/estimate_api_size.py` estimates the online dependency closure at
**~153 MB** (numpy ~54, onnxruntime ~48 being the largest), under the 250 MB
limit — so the dense path (which uses `fastembed`) should deploy. The authoritative check is still a
real Vercel build ([`scripts/size_check.md`](scripts/size_check.md)).

### Operational latency

Report Vercel **cold-start** and **warm-request** latency separately (hit
`/api/health` after idling, then again immediately) once deployed.

## 7. Security, privacy, guardrails

- **Secrets** live only in `.env` (gitignored) and Vercel env vars — never in
  code or in `index/`. If a key reaches git, **revoke it** (deleting the commit
  isn't enough).
- **Confidentiality:** `data/private/` is gitignored and never enters the index.
  The committed index is built from `data/public/` only. **Manually read
  `index/chunks.json` before every commit** — it's the one file that ships.
- **Prompt injection:** retrieved text is treated as untrusted data. The system
  prompt instructs the model to ignore instructions inside sources, and
  malicious text is kept in the data block, never the system role. Static tests
  in `tests/test_prompt_injection.py`; a behavioural corpus in
  `tests/fixtures/injection/` for manual checks.
- **Abuse limits:** max question length (config `limits.max_question_chars`),
  request timeout, structured logging that records question *length* only —
  never content or secrets.

## 8. Limitations & honest caveats

- Live function is read-only: adding documents means re-running the offline
  pipeline and redeploying — not a live upload feature.
- fastembed query embeddings (ONNX) vs sentence-transformers index embeddings
  (PyTorch) are the same model but not bit-identical; any tiny retrieval
  difference is documented by the eval, not hidden.
- **Refusal is primarily LLM-driven.** The pre-LLM confidence threshold is a
  cheap backstop, not the main gate. Measured on the sample corpus,
  `bge-small-en-v1.5` compresses cosine similarities into ~[0.40, 0.85] — even a
  clearly off-topic query ("how do I bake a cake?") scores ~0.46 — so a low
  absolute threshold rarely fires. The LLM, instructed to answer only from
  sources and otherwise say "I can't find that…", is what actually refuses
  unanswerable-but-on-topic questions (e.g. "what's the Wi-Fi password?"). The
  threshold is left conservative to avoid false refusals and is a **Week-5
  tuning task on the real corpus** (risk of overfitting a small set is noted).
- Cold starts can make the first request slow; a short demo video is recommended
  alongside the live URL.

## 9. Roadmap

Query rewriting for follow-ups · streaming responses · multi-turn memory with a
token budget · deploy the reranker if a size check shows it fits · compare two
embedding models · expose the retriever as a reusable package (reused by a later
customer-care agent project).

## 10. Repository layout

```
api/index.py            FastAPI serverless function (/api/ask, /api/health)
api/requirements.txt    SLIM online deps (the only ones deployed)
public/index.html       static frontend
config.yaml             chunking / retrieval / LLM settings
src/ingest/             parsers, clean, chunker, pipeline (offline)
src/retrieval/          bm25, hybrid, engine (shared) + embedders, chroma, reranker
src/generation/         prompts, llm (shared)
src/eval/               dataset.jsonl + retrieval/answer/ablation harness
tests/                  unit tests + injection fixtures
scripts/                query/ask CLIs, size-check helpers
.github/workflows/ci.yml  ruff + pytest on every push
```
