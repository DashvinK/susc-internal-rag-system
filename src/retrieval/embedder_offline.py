"""Offline embedder -- sentence-transformers (full PyTorch, CPU).

Used ONLY by the ingestion pipeline and the ablation study. Never deployed.
Builds the ``vectors.npy`` that the live function later loads.

Passages are embedded WITHOUT the query prefix; queries (used in the offline
ablations) are embedded WITH it, matching how bge models are trained and how
``embedder_online.embed_query`` behaves in production.
"""

from __future__ import annotations

import functools

import numpy as np

from src import config


@functools.lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.get("embedding.offline_model"))


def embed_passages(passages: list[str], batch_size: int = 32) -> np.ndarray:
    model = _model()
    vecs = model.encode(
        passages,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=config.get("embedding.normalize", True),
        show_progress_bar=len(passages) > 64,
    )
    return np.asarray(vecs, dtype=np.float32)


def embed_query(question: str) -> np.ndarray:
    model = _model()
    text = config.get("embedding.query_prefix", "") + question
    vec = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=config.get("embedding.normalize", True),
    )[0]
    return np.asarray(vec, dtype=np.float32)
