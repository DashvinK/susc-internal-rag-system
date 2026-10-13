"""RagEngine end-to-end with a mocked LLM and a tiny in-memory index (BM25 mode).

No network, no API key, no fastembed -- exercises exactly the code path the
deployed function runs, minus the real model call.
"""

from src.generation import prompts
from src.retrieval import bm25 as bm25_mod
from src.retrieval.engine import RagEngine

CHUNKS = [
    {
        "id": "a",
        "text": "EMR stands for Executive Member Recruitment.",
        "source": "emr.md",
        "section": "Overview",
        "chunk_index": 0,
    },
    {
        "id": "b",
        "text": "Orientation is held in the first week of each semester.",
        "source": "orientation.md",
        "section": "Schedule",
        "chunk_index": 0,
    },
    {
        "id": "c",
        "text": "Reimbursement claims must be submitted within thirty days.",
        "source": "sop.md",
        "section": "Claims",
        "chunk_index": 0,
    },
]


def make_engine(bm25_threshold: float = 0.0) -> RagEngine:
    tokenized = [bm25_mod.tokenize(c["text"]) for c in CHUNKS]
    eng = RagEngine(chunks=CHUNKS, tokenized_corpus=tokenized, vectors=None, mode="bm25")
    eng.bm25_refusal_threshold = bm25_threshold
    return eng


def test_answer_calls_llm_and_returns_cited_sources(monkeypatch):
    monkeypatch.setattr(
        "src.generation.llm.chat",
        lambda *a, **k: "EMR means Executive Member Recruitment [1].",
    )
    out = make_engine(bm25_threshold=0.0).answer("What does EMR stand for?")
    assert out["refused"] is False
    assert out["sources"], "expected at least one cited source"
    assert out["sources"][0]["source"] == "emr.md"
    assert "[1]" in out["answer"]
    assert set(out["timings_ms"]) == {"retrieve", "generate", "total"}


def test_low_confidence_refuses_without_calling_llm(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("LLM must not be called when retrieval is below threshold")

    monkeypatch.setattr("src.generation.llm.chat", boom)
    out = make_engine(bm25_threshold=1e9).answer("What does EMR stand for?")
    assert out["refused"] is True
    assert out["answer"] == prompts.REFUSAL_TEXT
    assert out["sources"] == []
    assert out["timings_ms"]["generate"] == 0.0


def test_model_refusal_is_propagated(monkeypatch):
    monkeypatch.setattr("src.generation.llm.chat", lambda *a, **k: prompts.REFUSAL_TEXT)
    out = make_engine(bm25_threshold=0.0).answer("What does EMR stand for?")
    assert out["refused"] is True
    assert out["sources"] == []
