"""Local retrieval sanity check (no LLM, no API key).

Prints the top-k retrieved chunks with scores and sources so you can judge by
eye whether the answer to a question is actually in the retrieved context --
the Week-2 "done when" check.

Run:
    python scripts/query_cli.py "What does EMR stand for?"
    python scripts/query_cli.py --mode bm25 "event proposal deadline"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.engine import RagEngine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="+", help="The question to retrieve for")
    parser.add_argument("--mode", default=None, help="hybrid | dense | bm25")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    question = " ".join(args.question)
    engine = RagEngine.load_from_dir(mode=args.mode)
    engine.top_k = args.k
    results, meta = engine.retrieve(question)

    print(f"\nQ: {question}")
    print(
        f"mode={meta['effective_mode']}  confidence={meta['confidence']}  "
        f"threshold={meta['threshold']}  fell_back={meta['fell_back_to_bm25']}\n"
    )
    for i, r in enumerate(results, 1):
        preview = r["text"].replace("\n", " ")[:160]
        print(f"[{i}] {r['source']} > {r['section']}")
        print(f"    score={r['score']} dense={r['dense_score']} bm25={r['bm25_score']}")
        print(f"    {preview}...\n")


if __name__ == "__main__":
    main()
