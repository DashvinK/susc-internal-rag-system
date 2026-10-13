"""Retrieval evaluation: hit@k and MRR against hand-labelled gold sources.

Measures ONLY retrieval (no LLM, no API key needed), so it is fast and cheap to
run repeatedly. Evaluates the deployed configuration by default (RagEngine with
the committed index + fastembed queries).

Run:
    python -m src.eval.retrieval_eval
    python -m src.eval.retrieval_eval --mode bm25
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from src import config
from src.retrieval.engine import RagEngine

DATASET = Path(__file__).resolve().parent / "dataset.jsonl"
RUNS_DIR = Path(__file__).resolve().parent / "runs"


def load_dataset(path: Path = DATASET) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _gold_set(item: dict) -> set[tuple[str, str]]:
    return {(g["file"], g["section"]) for g in item["gold_sources"]}


def _retrieved_keys(results: list[dict]) -> list[tuple[str, str]]:
    return [(r["source"], r["section"]) for r in results]


def evaluate_retrieval(engine: RagEngine, dataset: list[dict], ks=(1, 3, 5)) -> dict:
    answerable = [d for d in dataset if d["gold_sources"]]
    per_q = []
    hits = {k: 0 for k in ks}
    rr_sum = 0.0

    for item in answerable:
        results, _meta = engine.retrieve(item["question"])
        retrieved = _retrieved_keys(results)
        gold = _gold_set(item)

        # First rank (1-based) at which any gold source appears.
        first_rank = None
        for rank, key in enumerate(retrieved, start=1):
            if key in gold:
                first_rank = rank
                break

        for k in ks:
            if first_rank is not None and first_rank <= k:
                hits[k] += 1
        rr = (1.0 / first_rank) if first_rank else 0.0
        rr_sum += rr
        per_q.append(
            {"id": item["id"], "type": item["type"], "first_rank": first_rank, "rr": round(rr, 4)}
        )

    n = len(answerable)
    metrics = {f"hit@{k}": round(hits[k] / n, 4) for k in ks}
    metrics["mrr"] = round(rr_sum / n, 4)
    metrics["n_answerable"] = n
    return {"metrics": metrics, "per_question": per_q}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=None, help="hybrid | dense | bm25 (default: config)")
    parser.add_argument("--save", action="store_true", help="Save the run under src/eval/runs/")
    args = parser.parse_args()

    engine = RagEngine.load_from_dir(mode=args.mode)
    dataset = load_dataset()
    report = evaluate_retrieval(engine, dataset)

    print(f"\n=== Retrieval evaluation (mode={engine.mode}) ===")
    for key, val in report["metrics"].items():
        print(f"  {key:12s}: {val}")

    misses = [p for p in report["per_question"] if p["first_rank"] is None]
    if misses:
        print(
            f"\n  Misses (no gold source in top-{engine.top_k}): "
            + ", ".join(p["id"] for p in misses)
        )

    if args.save:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = RUNS_DIR / f"{stamp}_retrieval_{engine.mode}.json"
        out.write_text(
            json.dumps(
                {"mode": engine.mode, "config": config.get("retrieval"), **report},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n  Saved -> {out}")


if __name__ == "__main__":
    main()
