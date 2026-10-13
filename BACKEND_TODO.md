# Backend — Everything You Need To Do

A sequential checklist to take the backend from "built & verified locally" to
"live on Vercel with your real SUSC corpus". Work top to bottom; each step says
what to run and how to know it worked.

> **Current state:** code is complete, 25 tests pass, `ruff` is clean, and the
> pipeline + API + eval all run on the **sample** corpus. What remains is your
> content, your API key, your evaluation numbers, and the deploy.

Legend: 🟢 = already done for you · 🟡 = you must do it · ⏱️ = rough time.

---

## 0. Prerequisites (verify) 🟢/🟡

- [x] 🟢 Python 3.12 virtual environment at `.venv/`
- [x] 🟢 Offline dependencies installed (`requirements.txt`, minus optional chromadb)
- [x] 🟢 Index built from the sample corpus (`index/` has 4 files)
- [ ] 🟡 Activate the venv in every new terminal before running anything:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
  (Git Bash: `source .venv/Scripts/activate`)

If you ever move to a fresh machine: `python -m venv .venv` then
`pip install -r requirements.txt`.

---

## 1. Get a Groq API key & create `.env` 🟡 ⏱️5 min

The backend cannot generate answers without an LLM key. Groq is free, email-only,
no card.

1. [ ] Create a key at <https://console.groq.com/keys>.
2. [ ] Copy the example env file and paste your key in:
   ```powershell
   Copy-Item .env.example .env
   ```
   Then edit `.env` and set:
   ```
   GROQ_API_KEY=gsk_...your real key...
   ```
3. [ ] Confirm the current model id is still live at
   <https://console.groq.com/docs/models>. If `llama-3.3-70b-versatile` was
   retired, set a replacement in `.env`:
   ```
   LLM_MODEL=<current-model-id>
   ```

**Never commit `.env`.** It is gitignored. If a key ever lands in git history,
**revoke it immediately** at the Groq console — deleting the commit is not enough.

**Verify:**
```powershell
python scripts/ask_cli.py "What does EMR stand for?"
```
You should get a JSON answer with `"refused": false` and a cited source. A `401`
means the key is wrong; a `KeyError`/missing key means `.env` wasn't loaded.

---

## 2. Replace the sample corpus with real SUSC docs 🟡 ⏱️30–60 min

The documents in `data/public/` are **fictional placeholders**. Replace them.

1. [ ] Delete the four sample files in `data/public/` (keep `README.md`).
2. [ ] Add your real, **shareable** SUSC documents to `data/public/`.
       Supported: `.pdf`, `.docx`, `.md`, `.txt`. Filenames become citation keys,
       so keep them human-readable and unique (e.g. `constitution_2026.pdf`).
3. [ ] Put anything confidential (internal minutes, member contact details,
       anything you hold as Director of Extracurricular that isn't public) in
       `data/private/` instead — it is **gitignored and never indexed**.
4. [ ] Get sign-off from any co-owners of documents (e.g. co-signed EMR
       materials) before anything goes into a public repo.

> Rule of thumb: if you'd hesitate to post it on a public website, it goes in
> `data/private/`, not `data/public/`.

---

## 3. Rebuild the index 🟡 ⏱️2 min

The index is the only thing that ships to production, so rebuild it whenever the
corpus changes.

```powershell
python -m src.ingest.pipeline --full
```

- `--full` forces a complete rebuild (use it after swapping the whole corpus).
- Without `--full` it is **incremental**: only changed/added/removed files are
  re-embedded (fast for small edits).

This writes/overwrites `index/chunks.json`, `index/vectors.npy`,
`index/bm25_corpus.json`, `index/manifest.json`.

**Verify:** the summary prints your real file count and chunk count.

### 🔒 Mandatory privacy check before you commit

`index/chunks.json` is the one file that actually ships. **Read it** and confirm
nothing confidential leaked in:
```powershell
code index/chunks.json      # or: notepad index\chunks.json
```
Do this **every time** before `git commit`.

---

## 4. Sanity-check retrieval & answering locally 🟡 ⏱️10 min

Before trusting metrics, eyeball a few queries.

```powershell
# Retrieval only (no LLM, no key needed) — are the right chunks coming back?
python scripts/query_cli.py "a real question about your docs"

# End-to-end (uses the LLM) — is the answer grounded and cited?
python scripts/ask_cli.py "another real question"
```

- [ ] For ~10 questions, confirm the top chunks actually contain the answer.
- [ ] Confirm an obviously-absent fact triggers a refusal
      ("I can't find that in the SUSC documents.").

Also run the API locally and hit it directly:
```powershell
.venv\Scripts\uvicorn api.index:app --reload --port 8000
```
```bash
curl -X POST localhost:8000/api/ask -H "content-type: application/json" -d "{\"question\":\"...\"}"
curl localhost:8000/api/health
```

---

## 5. Build the evaluation set 🟡 ⏱️2–3 h

Edit `src/eval/dataset.jsonl`. It currently has 18 scaffold questions against the
sample corpus — **replace them** with questions grounded in YOUR docs.

Target **≥ 40 questions**, including **≥ 8 unanswerable** ones. Each line is one
JSON object:
```json
{"id": "q01", "question": "...", "gold_answer": "...", "gold_sources": [{"file": "constitution_2026.pdf", "section": "Article 3 — ..."}], "type": "factual"}
```
- `type` ∈ `factual`, `keyword`, `paraphrase`, `multi`, `unanswerable`.
- `gold_sources` must match the `source` filename and `section` heading exactly
  as they appear in `index/chunks.json` (open it to copy the real section text).
- For `unanswerable`, set `gold_sources: []` and
  `gold_answer: "I can't find that in the SUSC documents."`.

Suggested mix (from the build plan): 12 factual, 8 keyword/acronym, 8 paraphrase,
6 multi-chunk, 8+ unanswerable.

---

## 6. Run the evaluation & record numbers 🟡 ⏱️30 min

```powershell
# Retrieval metrics (no key needed) — saves a run under src/eval/runs/
python -m src.eval.retrieval_eval --save
python -m src.eval.retrieval_eval --mode dense
python -m src.eval.retrieval_eval --mode bm25

# Answer quality (needs GROQ_API_KEY). --no-judge skips the LLM judge.
python -m src.eval.answer_eval --save
```

- [ ] Paste the real hit@1/3/5, MRR, citation rate, and refusal accuracy into the
      **Baseline results** table in `README.md`.
- [ ] Manually grade ~20 answers yourself and compare with the LLM judge; note the
      agreement % in the README (judge hygiene).

---

## 7. Run the ablation study 🟡 ⏱️20 min

```powershell
python -m src.eval.run_ablations
# skip the heavy cross-encoder if it's slow on your machine:
python -m src.eval.run_ablations --no-rerank
```

- [ ] Copy the generated table from `src/eval/runs/ablations_*.md` into the
      **Ablation table** in `README.md`.
- [ ] Write one honest paragraph: does hybrid capture most of the benefit, or
      would the reranker be worth deploying later? Mark which row is deployed.

---

## 8. Tune the refusal threshold 🟡 ⏱️20 min

The pre-LLM backstop lives in `config.yaml`:
```yaml
retrieval:
  refusal_threshold: 0.30        # cosine; raise to refuse more aggressively
  bm25_refusal_threshold: 2.0
```
- [ ] Using your eval set, find a threshold that refuses the unanswerable
      questions without refusing real ones. (Note: `bge-small` compresses cosine
      into ~0.40–0.85, so the LLM is still the primary refusal gate — see the
      README limitation.)
- [ ] You can override it in production without a redeploy via the
      `REFUSAL_THRESHOLD` env var in Vercel.

---

## 9. Prompt-injection check 🟡 ⏱️10 min

```powershell
python -m pytest tests/test_prompt_injection.py      # static guarantees (CI)
```
- [ ] Behavioural check with the real LLM: follow
      `tests/fixtures/injection/README.md` and confirm the assistant ignores
      instructions embedded in documents (doesn't reveal its prompt, doesn't adopt
      planted "facts").

---

## 10. Deploy to Vercel 🟡 ⏱️30 min

1. [ ] Push the repo to GitHub (public):
   ```powershell
   git remote add origin https://github.com/<you>/susc-rag.git
   git push -u origin main
   ```
   (CI runs automatically on push — confirm the Actions tab is green.)
2. [ ] At <https://vercel.com>, sign in with GitHub, **Import** the repo. No card
       for Hobby.
3. [ ] In Vercel → Project → Settings → **Environment Variables**, add:
   - `GROQ_API_KEY` = your key
   - `LLM_MODEL` = current Groq model id
   - (optional) `RETRIEVAL_MODE` = `hybrid` (or `bm25` as the fallback)
   - (optional) `REFUSAL_THRESHOLD`
4. [ ] Deploy. `vercel.json` routes `/api/*` to the function and serves `public/`.
       Only **`api/requirements.txt`** (slim) affects the function build.
5. [ ] **Size check (do this now, and after every new API dependency):**
   ```powershell
   python scripts/estimate_api_size.py        # local proxy; ~153 MB currently
   ```
   The authoritative test is the real Vercel build — see `scripts/size_check.md`.
   If the build fails with a size error, set `RETRIEVAL_MODE=bm25` in Vercel; the
   engine auto-degrades to keyword-only and still works.

---

## 11. Post-deploy verification 🟡 ⏱️15 min

- [ ] Open the Vercel URL and ask a real question in the browser.
- [ ] Hit `/api/health` → expect `{"status":"ok", ...}`.
- [ ] Measure **cold vs warm** latency and record both in the README:
  ```bash
  # cold: after idling a few minutes
  curl -w "%{time_total}s\n" -o NUL -s https://<you>.vercel.app/api/health
  # warm: immediately again
  curl -w "%{time_total}s\n" -o NUL -s https://<you>.vercel.app/api/health
  ```
- [ ] Run the answer eval against the live endpoint:
  ```powershell
  python -m src.eval.answer_eval --endpoint https://<you>.vercel.app
  ```
- [ ] Add the live link to the top of `README.md`.
- [ ] Record a short demo video (cold starts can make the first interview request
      feel slow; the video shows it at its best).

---

## 12. Security checklist (every commit) 🟡

- [ ] `index/chunks.json` read through — nothing confidential.
- [ ] No real `.env` staged (`git status` should never show `.env`).
- [ ] No secrets in code or logs (logging records question *length* only).
- [ ] `data/private/` contents are not staged (gitignored, but double-check).

---

## Command cheat-sheet

| Goal | Command |
|---|---|
| Rebuild index (full) | `python -m src.ingest.pipeline --full` |
| Rebuild index (incremental) | `python -m src.ingest.pipeline` |
| Retrieval sanity (no key) | `python scripts/query_cli.py "..."` |
| Ask end-to-end (needs key) | `python scripts/ask_cli.py "..."` |
| Run API locally | `.venv\Scripts\uvicorn api.index:app --reload --port 8000` |
| Retrieval metrics | `python -m src.eval.retrieval_eval --save` |
| Answer metrics | `python -m src.eval.answer_eval --save` |
| Ablation study | `python -m src.eval.run_ablations` |
| Size estimate | `python scripts/estimate_api_size.py` |
| Lint + format | `ruff check . ; ruff format .` |
| Tests | `pytest` |

---

## Troubleshooting

| Symptom | Likely cause → fix |
|---|---|
| `401 Invalid API Key` | `GROQ_API_KEY` missing/wrong in `.env`, or not loaded — check `.env` exists and the key is valid. |
| `ModuleNotFoundError` | venv not activated, or deps not installed (`pip install -r requirements.txt`). |
| Pipeline: "No chunks produced" | `data/public/` is empty or only has unsupported file types. |
| Index mismatch error on API start | `vectors.npy` and `chunks.json` out of sync → rebuild with `--full`. |
| Vercel build fails on size | `fastembed` too big → set `RETRIEVAL_MODE=bm25` in Vercel env vars. |
| `chromadb` install fails on Windows | It's optional; you don't need it. Install MS C++ Build Tools only if you want `requirements-optional.txt`. |
| Retrieval returns wrong chunks | Check chunk size/overlap in `config.yaml`; eyeball with `query_cli.py`; run the ablation study. |
| Rate-limited during eval | The LLM judge caches responses in `.llm_cache/`; re-run to resume. Backoff is automatic. |
