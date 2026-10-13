"""API schema validation + a mocked-LLM request through the FastAPI app."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import api.index as api_index
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
    # A third chunk is needed so BM25's IDF stays positive: with only two docs,
    # a term appearing in exactly one of them gets IDF log(1.5/1.5)=0.
    {
        "id": "c",
        "text": "Reimbursement claims must be submitted within thirty days.",
        "source": "sop.md",
        "section": "Claims",
        "chunk_index": 0,
    },
]


def _tiny_engine() -> RagEngine:
    tokenized = [bm25_mod.tokenize(c["text"]) for c in CHUNKS]
    eng = RagEngine(chunks=CHUNKS, tokenized_corpus=tokenized, vectors=None, mode="bm25")
    eng.bm25_refusal_threshold = 0.0
    return eng


def test_ask_request_rejects_empty():
    with pytest.raises(ValidationError):
        api_index.AskRequest(question="")


def test_ask_request_rejects_too_long():
    with pytest.raises(ValidationError):
        api_index.AskRequest(question="x" * (api_index._MAX_Q + 1))


def test_health_and_ask_roundtrip(monkeypatch):
    api_index._engine = _tiny_engine()  # inject so get_engine() returns our tiny index
    monkeypatch.setattr(
        "src.generation.llm.chat",
        lambda *a, **k: "EMR means Executive Member Recruitment [1].",
    )
    client = TestClient(api_index.app)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    resp = client.post("/api/ask", json={"question": "What does EMR stand for?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is False
    assert "[1]" in body["answer"]
    assert body["sources"][0]["source"] == "emr.md"
    assert {"retrieve", "generate", "total"} <= set(body["timings_ms"])


def test_ask_rejects_empty_body(monkeypatch):
    api_index._engine = _tiny_engine()
    client = TestClient(api_index.app)
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
