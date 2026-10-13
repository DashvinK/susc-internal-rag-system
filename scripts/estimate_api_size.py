"""Rough local estimate of the deployed function's installed size.

Sums the on-disk size of the online dependencies in the CURRENT environment.
This is a proxy only -- Vercel installs Linux wheels that differ from your
local ones -- but it flags an obvious size problem early (Week 1).

Run inside the venv where api/requirements.txt is installed:
    python scripts/estimate_api_size.py
"""

from __future__ import annotations

import importlib.metadata as md

# Top-level distributions in api/requirements.txt (their transitive deps are
# counted too, via each distribution's recorded files).
ONLINE_DISTS = [
    "fastapi",
    "pydantic",
    "openai",
    "numpy",
    "rank-bm25",
    "fastembed",
    "PyYAML",
]
LIMIT_MB = 250


def dist_size_bytes(dist_name: str) -> int:
    try:
        dist = md.distribution(dist_name)
    except md.PackageNotFoundError:
        return -1
    total = 0
    for file in dist.files or []:
        try:
            total += (dist.locate_file(file)).stat().st_size
        except OSError:
            pass
    return total


def _runtime_dependencies(name: str) -> list[str]:
    """Return the dependency names pip would actually install for ``name``.

    Crucially, this SKIPS requirements gated behind an optional extra (markers
    like ``; extra == "torch"``) and markers that are false in this environment.
    Without this, transformers' huge optional backends (jax, paddlepaddle, ...)
    get counted even though pip never installs them -- wildly overestimating size.
    """
    from packaging.requirements import Requirement

    try:
        reqs = md.distribution(name).requires or []
    except md.PackageNotFoundError:
        return []
    deps = []
    for raw in reqs:
        try:
            req = Requirement(raw)
        except Exception:  # noqa: BLE001
            continue
        # Evaluate with NO extras selected; skip anything that isn't pulled in.
        if req.marker is not None and not req.marker.evaluate({"extra": ""}):
            continue
        deps.append(req.name)
    return deps


def main() -> None:
    seen: dict[str, int] = {}
    # Walk the runtime dependency closure so transitive packages (onnxruntime,
    # tokenizers, huggingface-hub, starlette, httpx, ...) are included -- but
    # NOT optional extras.
    frontier = list(ONLINE_DISTS)
    while frontier:
        name = frontier.pop()
        key = name.lower().replace("_", "-")
        if key in seen:
            continue
        seen[key] = dist_size_bytes(name)
        for dep in _runtime_dependencies(name):
            if dep.lower().replace("_", "-") not in seen:
                frontier.append(dep)

    total = sum(v for v in seen.values() if v > 0)
    print(f"\n{'package':30s} {'size (MB)':>12s}")
    print("-" * 44)
    for name, size in sorted(seen.items(), key=lambda kv: kv[1], reverse=True):
        tag = f"{size / 1e6:12.1f}" if size >= 0 else "   (missing)"
        print(f"{name:30s} {tag}")
    print("-" * 44)
    print(f"{'TOTAL':30s} {total / 1e6:12.1f}")
    print(f"\nVercel unzipped limit: {LIMIT_MB} MB")
    if total / 1e6 > LIMIT_MB * 0.8:
        print("WARNING: close to or over the limit -- plan for the BM25-only fallback.")
    else:
        print("OK locally. Still run the authoritative deploy test (scripts/size_check.md).")


if __name__ == "__main__":
    main()
