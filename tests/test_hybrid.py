"""RRF math and cosine top-k correctness."""

import numpy as np

from src.retrieval.hybrid import cosine_top_k, reciprocal_rank_fusion


def test_rrf_rewards_agreement_across_lists():
    # Doc 2 is ranked highly by BOTH retrievers; doc 0 and 3 only by one each.
    dense = [2, 0, 1]
    bm25 = [2, 3, 1]
    fused = reciprocal_rank_fusion([dense, bm25], k=60)
    assert fused[0][0] == 2  # agreed-upon doc wins


def test_rrf_exact_scores():
    # Single list: score for rank r (0-based) is 1/(k+r+1).
    fused = dict(reciprocal_rank_fusion([[5, 7]], k=60))
    assert abs(fused[5] - 1 / 61) < 1e-9
    assert abs(fused[7] - 1 / 62) < 1e-9


def test_rrf_top_k_truncates():
    fused = reciprocal_rank_fusion([[0, 1, 2, 3, 4]], k=60, top_k=2)
    assert len(fused) == 2
    assert [i for i, _ in fused] == [0, 1]


def test_cosine_top_k_orders_by_similarity():
    # Normalized vectors; query closest to row 1.
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    query = np.array([0.0, 1.0], dtype=np.float32)
    top = cosine_top_k(query, matrix, k=2)
    assert top[0][0] == 1
    assert abs(top[0][1] - 1.0) < 1e-6


def test_cosine_top_k_empty_matrix():
    assert (
        cosine_top_k(np.array([1.0, 0.0], dtype=np.float32), np.empty((0, 2), np.float32), k=3)
        == []
    )
