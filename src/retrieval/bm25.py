"""BM25 keyword retrieval -- shared by the offline eval AND the live API.

BM25 catches exact terms (SUSC acronyms, role names, dates) that dense
embeddings tend to blur. It runs in memory from a small bundled JSON corpus of
pre-tokenized chunks, so the serverless function needs no model and no DB.

Dependency-light on purpose: only ``numpy`` + ``rank_bm25``.
"""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

# Lowercase alphanumeric tokens. Simple and deterministic -- the SAME tokenizer
# must be used to build the corpus (offline) and to tokenize queries (online),
# which is exactly why this lives in one shared module.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_bm25(tokenized_corpus: list[list[str]]) -> BM25Okapi:
    return BM25Okapi(tokenized_corpus)


def score(bm25: BM25Okapi, query: str) -> np.ndarray:
    """Return a BM25 score per document, aligned to corpus order."""
    return np.asarray(bm25.get_scores(tokenize(query)), dtype=np.float32)


def top_k(bm25: BM25Okapi, query: str, k: int) -> list[tuple[int, float]]:
    """Return the top-k ``(doc_index, score)`` pairs, highest score first."""
    scores = score(bm25, query)
    if scores.size == 0:
        return []
    k = min(k, scores.size)
    # argpartition for the top-k, then sort just those by score descending.
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return [(int(i), float(scores[i])) for i in idx]
