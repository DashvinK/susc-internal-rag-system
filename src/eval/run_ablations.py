"""Ablation study (LOCAL ONLY) -- produces the README's results table.

Rebuilds the index from data/public at several chunk sizes and compares
dense-only vs BM25-only vs hybrid, with and without the cross-encoder reranker.
Uses the full offline tooling (sentence-transformers, optional cross-encoder);
retrieval is done in memory with NumPy, so ChromaDB is not required here.

Retrieval-only metrics (hit@3, MRR, p50 retrieve latency) by default so it runs
without an API key and without hitting LLM rate limits.

Run:
    python -m src.eval.run_ablations
    python -m src.eval.run_ablations --no-rerank     # skip the heavy cross-encoder
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import config
from src.eval.retrieval_eval import load_dataset
from src.ingest import clean
from src.ingest.chunker import Chunk, chunk_pages
from src.ingest.parsers import parse_directory
from src.retrieval import bm25 as bm25_mod
from src.retrieval import hybrid

RUNS_DIR = Path(__file__).resolve().parent / "runs"


@dataclass
class BuiltIndex:
    chunks: list[Chunk]
    vectors: np.ndarray
    bm25: object
    tokenized: list[list[str]]


def build(chunk_size: int, overlap: int) -> BuiltIndex:
    from src.retrieval import embedder_offline

    source_dir = config.resolve_path("paths.public_data")
    pages = parse_directory(source_dir)
    pages = [p for p in pages if p.source.lower() not in {"readme.md"}]
    for p in pages:
        p.text = clean.clean_text(p.text)
    pages = [p for p in pages if p.text]
    chunks = chunk_pages(pages, chunk_size, overlap)
    vectors = embedder_offline.embed_passages([c.text for c in chunks])
    tokenized = [bm25_mod.tokenize(c.text) for c in chunks]
    return BuiltIndex(chunks, vectors, bm25_mod.build_bm25(tokenized), tokenized)


def _rank(idx: BuiltIndex, question: str, mode: str, candidate_k: int) -> list[int]:
    from src.retrieval import embedder_offline

    dense = bm = []
    if mode in ("dense", "hybrid"):
        qv = embedder_offline.embed_query(question)
        dense = [i for i, _ in hybrid.cosine_top_k(qv, idx.vectors, candidate_k)]
    if mode in ("bm25", "hybrid"):
        bm = [i for i, _ in bm25_mod.top_k(idx.bm25, question, candidate_k)]
    if mode == "dense":
        return dense
    if mode == "bm25":
        return bm
    return [
        i
        for i, _ in hybrid.reciprocal_rank_fusion([dense, bm], k=config.get("retrieval.rrf_k", 60))
    ]


def _apply_rerank(
    idx: BuiltIndex, question: str, candidate_ids: list[int], top_k: int
) -> list[int]:
    from src.retrieval import reranker

    candidates = [idx.chunks[i] for i in candidate_ids[: max(top_k * 4, 20)]]
    ranked = reranker.rerank(question, candidates, top_k=len(candidates))
    by_id = {
        c.id: i for i, c in zip(candidate_ids, (idx.chunks[j] for j in candidate_ids), strict=False)
    }
    return [by_id[c.id] for c, _ in ranked if c.id in by_id]


def evaluate(idx: BuiltIndex, dataset: list[dict], mode: str, rerank: bool, top_k: int) -> dict:
    answerable = [d for d in dataset if d["gold_sources"]]
    hit3 = 0
    rr_sum = 0.0
    latencies = []
    candidate_k = int(config.get("retrieval.candidate_k", 20))

    for item in answerable:
        t0 = time.perf_counter()
        ranked = _rank(idx, item["question"], mode, candidate_k)
        if rerank:
            ranked = _apply_rerank(idx, item["question"], ranked, top_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        keys = [(idx.chunks[i].source, idx.chunks[i].section) for i in ranked[:top_k]]
        gold = {(g["file"], g["section"]) for g in item["gold_sources"]}
        first_rank = next((r for r, k in enumerate(keys, 1) if k in gold), None)
        if first_rank and first_rank <= 3:
            hit3 += 1
        rr_sum += (1.0 / first_rank) if first_rank else 0.0

    n = len(answerable)
    return {
        "hit@3": round(hit3 / n, 4),
        "mrr": round(rr_sum / n, 4),
        "p50_retrieve_ms": round(statistics.median(latencies), 1) if latencies else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-rerank", action="store_true", help="Skip the cross-encoder rows")
    args = parser.parse_args()

    dataset = load_dataset()
    overlap = int(config.get("chunking.overlap", 200))
    top_k = int(config.get("retrieval.top_k", 5))

    # (label, chunk_size_chars, mode, rerank)
    grid = [
        ("A", 1200, "dense", False),
        ("B", 2000, "dense", False),
        ("C", 3200, "dense", False),
        ("D", 2000, "bm25", False),
        ("E", 2000, "hybrid", False),
    ]
    if not args.no_rerank:
        grid.append(("F", 2000, "hybrid", True))

    # Build each distinct chunk size once, reuse across modes.
    built: dict[int, BuiltIndex] = {}
    rows = []
    for label, cs, mode, rerank in grid:
        if cs not in built:
            print(f"Building index @ chunk_size={cs} ...")
            built[cs] = build(cs, overlap)
        m = evaluate(built[cs], dataset, mode, rerank, top_k)
        rows.append(
            {
                "config": label,
                "chunk_size": cs,
                "retrieval": mode,
                "rerank": "yes" if rerank else "no",
                **m,
            }
        )
        print(f"  {label}: cs={cs} {mode} rerank={rerank} -> {m}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RUNS_DIR / f"ablations_{stamp}.csv"
    md_path = RUNS_DIR / f"ablations_{stamp}.md"

    fields = ["config", "chunk_size", "retrieval", "rerank", "hit@3", "mrr", "p50_retrieve_ms"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    for r in rows:
        md_lines.append("| " + " | ".join(str(r[f]) for f in fields) + " |")
    md = "\n".join(md_lines) + "\n"
    md_path.write_text(md, encoding="utf-8")

    print(f"\nSaved:\n  {csv_path}\n  {md_path}\n")
    print(md)


if __name__ == "__main__":
    main()
