"""Local end-to-end ask (retrieval + LLM). Requires GROQ_API_KEY in .env.

Run:
    python scripts/ask_cli.py "Who chairs the EMR shortlisting process?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.retrieval.engine import RagEngine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="+")
    parser.add_argument("--mode", default=None)
    args = parser.parse_args()

    engine = RagEngine.load_from_dir(mode=args.mode)
    result = engine.answer(" ".join(args.question))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
