"""Cross-encoder reranker -- LOCAL ONLY, for the ablation study.

A cross-encoder scores (query, passage) pairs jointly, which is more accurate
than bi-encoder cosine similarity but too heavy to deploy. Used in Week 5 to
measure whether a reranker would be worth shipping later.
"""

from __future__ import annotations

import functools

from src import config
from src.ingest.chunker import Chunk


@functools.lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.get("reranker.model"))


def rerank(query: str, candidates: list[Chunk], top_k: int) -> list[tuple[Chunk, float]]:
    """Reorder candidate chunks by cross-encoder relevance, best first."""
    if not candidates:
        return []
    model = _model()
    pairs = [[query, c.text] for c in candidates]
    scores = model.predict(pairs)
    ranked = sorted(
        zip(candidates, (float(s) for s in scores), strict=False), key=lambda x: x[1], reverse=True
    )
    return ranked[:top_k]
