"""Central configuration loader.

Reads ``config.yaml`` once and layers a small set of environment-variable
overrides on top, so the live Vercel function can be re-tuned (model name,
retrieval mode, refusal threshold) WITHOUT a redeploy.

This module is intentionally dependency-light (only PyYAML) so it is safe to
import inside the slim serverless function.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml

# Repo root = parent of the ``src`` directory. Works both locally and when the
# ``src`` tree is bundled next to ``api/index.py`` on Vercel.
REPO_ROOT = Path(__file__).resolve().parent.parent


def _deep_get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


@functools.lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    """Load ``config.yaml`` and apply environment overrides (cached)."""
    cfg_path = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(cfg_path, encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    # --- Environment overrides (production tuning without a code change) ---
    llm = cfg.setdefault("llm", {})
    if os.getenv("LLM_MODEL"):
        llm["model"] = os.environ["LLM_MODEL"]
    if os.getenv("LLM_PROVIDER"):
        llm["provider"] = os.environ["LLM_PROVIDER"]
    if os.getenv("LLM_BASE_URL"):
        llm["base_url"] = os.environ["LLM_BASE_URL"]

    retr = cfg.setdefault("retrieval", {})
    if os.getenv("RETRIEVAL_MODE"):
        retr["mode"] = os.environ["RETRIEVAL_MODE"]
    if os.getenv("REFUSAL_THRESHOLD"):
        retr["refusal_threshold"] = float(os.environ["REFUSAL_THRESHOLD"])

    return cfg


def get(dotted: str, default: Any = None) -> Any:
    """Convenience accessor, e.g. ``config.get("retrieval.top_k", 5)``."""
    return _deep_get(load_config(), dotted, default)


def resolve_path(dotted_or_path: str) -> Path:
    """Resolve a config path value (or a literal relative path) against the repo root."""
    value = get(dotted_or_path, dotted_or_path)
    p = Path(value)
    return p if p.is_absolute() else (REPO_ROOT / p)


def groq_api_key() -> str | None:
    """API key is read from the environment ONLY -- never from config.yaml."""
    return os.getenv("GROQ_API_KEY")
