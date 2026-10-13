"""LLM client -- one ``chat()`` function, shared everywhere.

Uses the OpenAI SDK pointed at an OpenAI-compatible base URL, so the same code
talks to Groq (production) or a local Ollama server (dev only) by changing
config/env alone.

Features:
  * Exponential backoff with jitter on HTTP 429 (rate limit) and transient errors.
  * Optional on-disk response cache keyed by hash(model + messages), used by the
    eval harness to survive free-tier rate limits. The API path never caches
    (serverless filesystem is ephemeral and per-request freshness matters).
"""

from __future__ import annotations

import functools
import hashlib
import json
import random
import time
from pathlib import Path

from src import config


@functools.lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    api_key = config.groq_api_key() or "missing-key"
    base_url = config.get("llm.base_url")
    timeout = config.get("llm.request_timeout_s", 30)
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def _cache_key(model: str, messages: list[dict], temperature: float) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def chat(
    messages: list[dict],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    cache_dir: str | Path | None = None,
) -> str:
    """Send a chat completion and return the assistant's text.

    Raises the last exception if all retries are exhausted.
    """
    model = model or config.get("llm.model")
    temperature = config.get("llm.temperature", 0.0) if temperature is None else temperature
    max_tokens = max_tokens or config.get("llm.max_tokens", 1024)
    max_retries = config.get("llm.max_retries", 4)

    # --- Optional disk cache (eval only) ---
    cache_file: Path | None = None
    if cache_dir is not None:
        cdir = Path(cache_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        cache_file = _cache_path(cdir, _cache_key(model, messages, temperature))
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))["content"]

    client = _client()
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            if cache_file is not None:
                cache_file.write_text(json.dumps({"content": content}), encoding="utf-8")
            return content
        except Exception as exc:  # noqa: BLE001 -- retry on any transient failure
            last_exc = exc
            status = getattr(exc, "status_code", None)
            retriable = status in (429, 500, 502, 503, 504) or status is None
            if not retriable or attempt == max_retries - 1:
                break
            # Exponential backoff with jitter: 1s, 2s, 4s, ... (+/- up to 1s).
            sleep_s = (2**attempt) + random.random()
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc
