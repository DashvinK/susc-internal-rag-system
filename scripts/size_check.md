# Vercel 250 MB size check (do this in Week 1, not Week 6)

The deployed function must stay under **250 MB unzipped**. `fastembed` pulls
`onnxruntime`, which is the single biggest risk. Confirm it fits *before*
building on top of it.

## Option A — real deploy test (authoritative)

1. Temporarily replace `api/index.py` with a throwaway:

   ```python
   import fastembed  # noqa: F401
   from fastapi import FastAPI
   app = FastAPI()

   @app.get("/api/health")
   def health():
       return {"ok": True}
   ```

2. Keep the slim `api/requirements.txt` as-is.
3. `git push` (or `vercel deploy`). If the build fails with a size error, you
   need the **BM25-only production fallback** (set `RETRIEVAL_MODE=bm25` in
   Vercel env vars). Dense/hybrid still run in the local ablation study.
4. Restore the real `api/index.py`.

## Option B — local proxy estimate (quick, approximate)

Run `python scripts/estimate_api_size.py` after `pip install -r api/requirements.txt`
into a **clean** venv. It sums the installed size of the online dependencies.
This is only a proxy — Vercel's Linux wheels differ from your local ones — but a
result well under 250 MB is reassuring, and one near/over it is a clear warning.

## If it does not fit

Set `RETRIEVAL_MODE=bm25` in the Vercel dashboard. The engine auto-degrades to
BM25-only (it already falls back if `fastembed` import fails). Document the
trade-off in the README: dense/hybrid improved metrics locally but did not fit
the serverless limit, so the live demo runs BM25 + the same generation pipeline.
