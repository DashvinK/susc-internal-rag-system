"""BM25 tokenization and ranking."""

from src.retrieval import bm25 as bm25_mod


def test_tokenize_lowercases_and_splits_alnum():
    assert bm25_mod.tokenize("EMR stands for Executive-Member Recruitment!") == [
        "emr",
        "stands",
        "for",
        "executive",
        "member",
        "recruitment",
    ]


def test_tokenize_keeps_numbers():
    assert bm25_mod.tokenize("submit within 30 days") == ["submit", "within", "30", "days"]


def test_bm25_ranks_exact_term_match_first():
    corpus = [
        bm25_mod.tokenize("The orientation schedule is in the first week."),
        bm25_mod.tokenize("EMR stands for Executive Member Recruitment."),
        bm25_mod.tokenize("Reimbursement claims need receipts."),
    ]
    bm25 = bm25_mod.build_bm25(corpus)
    top = bm25_mod.top_k(bm25, "What does EMR stand for?", k=3)
    assert top[0][0] == 1  # the EMR document ranks first
    assert top[0][1] > 0


def test_top_k_respects_k_and_is_sorted_desc():
    corpus = [bm25_mod.tokenize(t) for t in ["alpha beta", "beta gamma", "gamma delta"]]
    bm25 = bm25_mod.build_bm25(corpus)
    top = bm25_mod.top_k(bm25, "beta", k=2)
    assert len(top) == 2
    assert top[0][1] >= top[1][1]
