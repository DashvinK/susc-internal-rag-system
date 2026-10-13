"""RagEngine -- the shared retrieve-and-answer core.

Loaded once per cold start by ``api/index.py`` and also constructed directly by
the test suite with tiny in-memory data. Keeping ALL the logic here (not in the
API file) is what lets the deployed behaviour be unit-tested without a server.

Dependency-light at import time: numpy, rank_bm25, the shared prompt/llm modules.
fastembed (dense query embedding) is imported lazily and only if needed, so a
BM25-only deployment never pays for it -- and if the dense path fails for any
reason, the engine automatically degrades to BM25-only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src import config
from src.generation import llm, prompts
from src.retrieval import bm25 as bm25_mod
from src.retrieval import hybrid


class RagEngine:
    def __init__(
        self,
        chunks: list[dict],
        tokenized_corpus: list[list[str]],
        vectors: np.ndarray | None = None,
        mode: str | None = None,
    ) -> None:
        self.chunks = chunks
        self.vectors = vectors
        self.bm25 = bm25_mod.build_bm25(tokenized_corpus)
        self.mode = mode or config.get("retrieval.mode", "hybrid")
        self.top_k = int(config.get("retrieval.top_k", 5))
        self.candidate_k = int(config.get("retrieval.candidate_k", 20))
        self.rrf_k = int(config.get("retrieval.rrf_k", 60))
        self.refusal_threshold = float(config.get("retrieval.refusal_threshold", 0.30))
        self.bm25_refusal_threshold = float(config.get("retrieval.bm25_refusal_threshold", 2.0))

    # ----------------------------------------------------------------- loading
    @classmethod
    def load_from_dir(
        cls, index_dir: str | Path | None = None, mode: str | None = None
    ) -> RagEngine:
        idx = Path(index_dir) if index_dir else config.resolve_path("paths.index_dir")
        chunks = json.loads((idx / "chunks.json").read_text(encoding="utf-8"))
        tokenized = json.loads((idx / "bm25_corpus.json").read_text(encoding="utf-8"))
        vectors_path = idx / "vectors.npy"
        vectors = np.load(vectors_path) if vectors_path.exists() else None
        if vectors is not None and vectors.shape[0] != len(chunks):
            raise ValueError(
                f"Index mismatch: {vectors.shape[0]} vectors vs {len(chunks)} chunks. "
                "Rebuild the index with `python -m src.ingest.pipeline`."
            )
        return cls(chunks=chunks, tokenized_corpus=tokenized, vectors=vectors, mode=mode)

    # --------------------------------------------------------------- retrieval
    def retrieve(self, question: str) -> tuple[list[dict], dict]:
        timings: dict[str, float] = {}
        want_dense = (
            self.mode in ("dense", "hybrid") and self.vectors is not None and self.vectors.size > 0
        )
        dense_results: list[tuple[int, float]] = []
        fell_back = False

        if want_dense:
            try:
                from src.retrieval import embedder_online

                t0 = time.perf_counter()
                qvec = embedder_online.embed_query(question)
                timings["embed"] = (time.perf_counter() - t0) * 1000
                dense_results = hybrid.cosine_top_k(qvec, self.vectors, self.candidate_k)
            except Exception:  # noqa: BLE001 -- any dense failure => BM25-only fallback
                want_dense = False
                fell_back = self.mode in ("dense", "hybrid")

        want_bm25 = (self.mode in ("bm25", "hybrid")) or (not want_dense)
        bm25_results: list[tuple[int, float]] = []
        if want_bm25:
            t0 = time.perf_counter()
            bm25_results = bm25_mod.top_k(self.bm25, question, self.candidate_k)
            timings["bm25"] = (time.perf_counter() - t0) * 1000

        if want_dense and want_bm25:
            effective_mode = "hybrid"
        elif want_dense:
            effective_mode = "dense"
        else:
            effective_mode = "bm25"

        # --- Fuse / select ---
        dense_scores = {i: s for i, s in dense_results}
        bm25_scores = {i: s for i, s in bm25_results}

        if effective_mode == "hybrid":
            fused = hybrid.reciprocal_rank_fusion(
                [[i for i, _ in dense_results], [i for i, _ in bm25_results]],
                k=self.rrf_k,
                top_k=self.top_k,
            )
        elif effective_mode == "dense":
            fused = dense_results[: self.top_k]
        else:
            fused = bm25_results[: self.top_k]

        results = []
        for idx, fused_score in fused:
            chunk = self.chunks[idx]
            results.append(
                {
                    "index": idx,
                    "source": chunk["source"],
                    "section": chunk["section"],
                    "text": chunk["text"],
                    "score": round(float(fused_score), 6),
                    "dense_score": round(float(dense_scores[idx]), 6)
                    if idx in dense_scores
                    else None,
                    "bm25_score": round(float(bm25_scores[idx]), 6) if idx in bm25_scores else None,
                }
            )

        # --- Confidence for refusal ---
        if want_dense and dense_results:
            confidence = max(s for _, s in dense_results)
            threshold = self.refusal_threshold
        elif bm25_results:
            confidence = bm25_results[0][1]
            threshold = self.bm25_refusal_threshold
        else:
            confidence = 0.0
            threshold = self.refusal_threshold

        meta = {
            "effective_mode": effective_mode,
            "fell_back_to_bm25": fell_back,
            "confidence": round(float(confidence), 6),
            "threshold": threshold,
            "timings": timings,
        }
        return results, meta

    # ------------------------------------------------------------------ answer
    def answer(self, question: str) -> dict:
        t_total = time.perf_counter()
        results, meta = self.retrieve(question)
        retrieve_ms = sum(meta["timings"].values())

        sources = [
            {
                "n": i + 1,
                "source": r["source"],
                "section": r["section"],
                "score": r["score"],
                "text": r["text"],
            }
            for i, r in enumerate(results)
        ]

        # Refuse WITHOUT calling the LLM when retrieval confidence is too low.
        if not results or meta["confidence"] < meta["threshold"]:
            return {
                "answer": prompts.REFUSAL_TEXT,
                "sources": [],
                "refused": True,
                "timings_ms": {
                    "retrieve": round(retrieve_ms, 1),
                    "generate": 0.0,
                    "total": round((time.perf_counter() - t_total) * 1000, 1),
                },
                "meta": meta,
            }

        messages = prompts.build_messages(question, sources)
        t_gen = time.perf_counter()
        answer_text = llm.chat(messages)
        generate_ms = (time.perf_counter() - t_gen) * 1000

        refused = (
            answer_text.strip().rstrip(".").lower() == prompts.REFUSAL_TEXT.rstrip(".").lower()
        )

        return {
            "answer": answer_text.strip(),
            # When the model itself refuses, don't present sources it didn't use.
            "sources": []
            if refused
            else [
                {"n": s["n"], "source": s["source"], "section": s["section"], "score": s["score"]}
                for s in sources
            ],
            "refused": refused,
            "timings_ms": {
                "retrieve": round(retrieve_ms, 1),
                "generate": round(generate_ms, 1),
                "total": round((time.perf_counter() - t_total) * 1000, 1),
            },
            "meta": meta,
        }
