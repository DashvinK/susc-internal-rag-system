"""Dense cosine retrieval + Reciprocal Rank Fusion (RRF).

Shared by the offline eval and the live API. Pure NumPy -- no model, no DB.

RRF fuses several ranked lists without needing their scores to be on the same
scale (cosine similarity and BM25 are not comparable numerically). Each list
contributes ``1 / (rrf_k + rank)`` to every doc it ranks; summing across lists
rewards docs that rank highly in more than one retriever.
"""

from __future__ import annotations

import numpy as np


def cosine_top_k(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Top-k by cosine similarity.

    Assumes rows of ``matrix`` and ``query_vec`` are L2-normalized (the pipeline
    normalizes them), so a dot product equals cosine similarity.
    """
    if matrix.size == 0:
        return []
    sims = matrix @ query_vec  # (N,) similarity per chunk
    k = min(k, sims.shape[0])
    idx = np.argpartition(sims, -k)[-k:]
    idx = idx[np.argsort(sims[idx])[::-1]]
    return [(int(i), float(sims[i])) for i in idx]


def reciprocal_rank_fusion(
    rankings: list[list[int]], k: int = 60, top_k: int | None = None
) -> list[tuple[int, float]]:
    """Fuse ranked lists of doc indices into one ranking.

    ``rankings`` is a list of lists; each inner list is doc indices ordered best
    first for one retriever. Returns ``(doc_index, rrf_score)`` best first.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return fused[:top_k] if top_k is not None else fused
