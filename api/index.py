"""Vercel serverless function: the live RAG API.

Exposes two routes:
    GET  /api/health  -> liveness + loaded-index summary
    POST /api/ask     -> {answer, sources[], refused, timings_ms}

Design notes
------------
* The index (chunks.json + vectors.npy + bm25_corpus.json) is loaded ONCE per
  cold start and cached at module scope via ``get_engine()`` -- never per request.
* Loading is lazy (first request), so importing this module has no side effects.
  That keeps unit tests fast and import-safe even without the index present.
* All real work lives in ``src.retrieval.engine.RagEngine`` so the deployed
  behaviour is exactly what the test suite exercises.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make the bundled ``src`` tree importable whether we run locally or on Vercel
# (where this file sits in api/ and src/ is bundled alongside via vercel.json).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src import config  # noqa: E402
from src.retrieval.engine import RagEngine  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("susc-rag")

app = FastAPI(title="SUSC Knowledge Assistant", version="1.0.0")

_engine: RagEngine | None = None


def get_engine() -> RagEngine:
    """Load (and cache) the RagEngine on first use."""
    global _engine
    if _engine is None:
        logger.info("Cold start: loading index into memory...")
        _engine = RagEngine.load_from_dir()
        logger.info("Index loaded: %d chunks, mode=%s", len(_engine.chunks), _engine.mode)
    return _engine


# --------------------------------------------------------------------- schemas
_MAX_Q = int(config.get("limits.max_question_chars", 1000))


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=_MAX_Q)


class SourceOut(BaseModel):
    n: int
    source: str
    section: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    refused: bool
    timings_ms: dict[str, float]


# ---------------------------------------------------------------------- routes
@app.get("/api/health")
def health() -> dict:
    try:
        engine = get_engine()
        return {
            "status": "ok",
            "chunks": len(engine.chunks),
            "mode": engine.mode,
            "model": config.get("llm.model"),
            "has_vectors": engine.vectors is not None,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Health check failed")
        return {"status": "error", "detail": str(exc)}


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    # Structured log: question LENGTH only, never the content or any secret.
    logger.info("ask: q_len=%d", len(question))

    try:
        result = get_engine().answer(question)
    except Exception as exc:  # noqa: BLE001
        logger.exception("answer() failed")
        raise HTTPException(status_code=502, detail="Upstream generation failed.") from exc

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        refused=result["refused"],
        timings_ms=result["timings_ms"],
    )
