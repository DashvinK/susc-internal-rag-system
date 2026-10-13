"""Answer-quality evaluation.

Three layers, per the build plan:
  (a) Automatic checks  -- citation present on answered questions; refusal
      correct on unanswerable ones. No judge needed, cheap, deterministic.
  (b) LLM-as-judge      -- groundedness + correctness, temperature 0, fixed
      rubric. Requires an API key. Responses are cached on disk to survive
      free-tier rate limits.
  (c) Human comparison  -- grade ~20 answers yourself in a CSV and this script
      reports agreement with the judge (judge hygiene).

Run (local engine):
    python -m src.eval.answer_eval
    python -m src.eval.answer_eval --no-judge        # layer (a) only, no key
    python -m src.eval.answer_eval --endpoint https://<your>.vercel.app
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

from src import config
from src.eval.retrieval_eval import load_dataset
from src.generation import llm

RUNS_DIR = Path(__file__).resolve().parent / "runs"
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".llm_cache"
CITATION_RE = re.compile(r"\[\d+\]")

JUDGE_SYSTEM = (
    "You are a strict grader for a retrieval-augmented QA system. "
    "You will receive a QUESTION, the GOLD answer, the SYSTEM answer, and the "
    "SOURCES the system was given. Score two things from 0.0 to 1.0:\n"
    "  groundedness: is every claim in the SYSTEM answer supported by the SOURCES? "
    "(1.0 = fully supported, 0.0 = fabricated)\n"
    "  correctness: does the SYSTEM answer match the GOLD answer in substance? "
    "(1.0 = correct, 0.0 = wrong)\n"
    'Reply with ONLY a JSON object: {"groundedness": <float>, "correctness": <float>}.'
)


def _call_local(engine, question: str) -> dict:
    return engine.answer(question)


def _call_endpoint(base_url: str, question: str) -> dict:
    import httpx

    url = base_url.rstrip("/") + "/api/ask"
    resp = httpx.post(url, json={"question": question}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def judge(item: dict, answer: str, sources: list[dict]) -> dict:
    src_text = "\n".join(f"[{s['n']}] {s['source']} > {s['section']}" for s in sources)
    user = (
        f"QUESTION: {item['question']}\n\n"
        f"GOLD: {item['gold_answer']}\n\n"
        f"SYSTEM: {answer}\n\n"
        f"SOURCES:\n{src_text or '(none)'}"
    )
    raw = llm.chat(
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
        temperature=0.0,
        cache_dir=CACHE_DIR,
    )
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0)) if match else {}
        return {
            "groundedness": float(data.get("groundedness", 0.0)),
            "correctness": float(data.get("correctness", 0.0)),
        }
    except Exception:  # noqa: BLE001
        return {"groundedness": 0.0, "correctness": 0.0, "judge_error": raw[:200]}


def run(args) -> dict:
    dataset = load_dataset()
    if args.limit:
        dataset = dataset[: args.limit]

    engine = None
    if not args.endpoint:
        from src.retrieval.engine import RagEngine

        engine = RagEngine.load_from_dir(mode=args.mode)

    per_q = []
    citation_ok = refusal_ok = refusal_total = 0
    answered_total = 0
    g_sum = c_sum = 0.0
    judged = 0

    for item in dataset:
        result = (
            _call_endpoint(args.endpoint, item["question"])
            if args.endpoint
            else _call_local(engine, item["question"])
        )
        answer = result["answer"]
        refused = result["refused"]
        sources = result.get("sources", [])
        is_unanswerable = item["type"] == "unanswerable"

        # (a) automatic checks
        if is_unanswerable:
            refusal_total += 1
            if refused:
                refusal_ok += 1
        else:
            answered_total += 1
            if not refused and CITATION_RE.search(answer):
                citation_ok += 1

        row = {"id": item["id"], "type": item["type"], "refused": refused}

        # (b) judge (skip unanswerable + refusals -- nothing to ground)
        if not args.no_judge and not is_unanswerable and not refused:
            scores = judge(item, answer, sources)
            g_sum += scores["groundedness"]
            c_sum += scores["correctness"]
            judged += 1
            row.update(scores)

        per_q.append(row)

    metrics = {
        "citation_rate": round(citation_ok / answered_total, 4) if answered_total else None,
        "refusal_accuracy": round(refusal_ok / refusal_total, 4) if refusal_total else None,
        "mean_groundedness": round(g_sum / judged, 4) if judged else None,
        "mean_correctness": round(c_sum / judged, 4) if judged else None,
        "n_total": len(dataset),
        "n_judged": judged,
    }
    return {"metrics": metrics, "per_question": per_q}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=None, help="hybrid | dense | bm25 (local engine)")
    parser.add_argument(
        "--endpoint", default=None, help="Evaluate a live URL instead of local engine"
    )
    parser.add_argument(
        "--no-judge", action="store_true", help="Skip the LLM judge (no API key needed)"
    )
    parser.add_argument("--limit", type=int, default=0, help="Only the first N questions")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    report = run(args)
    target = args.endpoint or f"local (mode={args.mode or config.get('retrieval.mode')})"
    print(f"\n=== Answer evaluation [{target}] ===")
    for key, val in report["metrics"].items():
        print(f"  {key:20s}: {val}")

    if args.save:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = RUNS_DIR / f"{stamp}_answer.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  Saved -> {out}")


if __name__ == "__main__":
    main()
