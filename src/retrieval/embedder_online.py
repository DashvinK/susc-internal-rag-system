"""Online query embedder -- fastembed (ONNX runtime, NO PyTorch).

This is what the live Vercel function uses to embed the incoming question. It
points at the SAME model as the offline index builder so the query vector lands
in the same space as the bundled chunk vectors.

The model is lazily loaded and cached on first use (downloaded once to a local
cache; on Vercel that cache lives in the writable /tmp). Keeping the import lazy
means a BM25-only deployment never pays the fastembed import cost.
"""

from __future__ import annotations

import functools

import numpy as np

from src import config


@functools.lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=config.get("embedding.online_model"))


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def embed_query(question: str) -> np.ndarray:
    """Embed a query using fastembed's query embedding (adds the bge instruction)."""
    model = _model()
    vec = np.asarray(next(iter(model.query_embed([question]))), dtype=np.float32)
    if config.get("embedding.normalize", True):
        vec = _normalize(vec)
    return vec


def embed_passages(passages: list[str]) -> np.ndarray:
    """Embed passages (no query prefix). Mainly for parity tests against the offline model."""
    model = _model()
    vecs = np.asarray(list(model.embed(passages)), dtype=np.float32)
    if config.get("embedding.normalize", True):
        vecs = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
    return vecs
