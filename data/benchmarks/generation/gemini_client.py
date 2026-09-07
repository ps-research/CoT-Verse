"""Gemini/Gemma API client with multi-key rotation and rate pacing.

Follows the exact API signature supplied by the author:
streaming generate_content_stream, thinking_config, googleSearch tool,
response_mime_type=application/json + response_schema.

Keys: one per line in the file named by $GEMINI_KEYS_FILE (fallback: the
single $GEMINI_API_KEY). Each key is rate-limited to 30 requests/minute and
14.4k/day — the pool paces each key to >=2s between calls and rotates to the
next key on quota errors (60s cooldown for the throttled key).
"""
import json
import os
import threading
import time
from pathlib import Path

from google import genai
from google.genai import types

DEFAULT_MODEL = "gemma-4-31b-it"   # author-mandated
PER_KEY_MIN_INTERVAL = 2.1         # 30 req/min per key
QUOTA_COOLDOWN = 60.0


class _KeyPool:
    def __init__(self):
        keys_file = os.environ.get("GEMINI_KEYS_FILE")
        if keys_file and Path(keys_file).exists():
            keys = [k.strip() for k in Path(keys_file).read_text().split() if k.strip()]
        else:
            single = os.environ.get("GEMINI_API_KEY")
            keys = [single] if single else []
        keys = list(dict.fromkeys(keys))   # dedupe: duplicate keys add no quota
        if not keys:
            raise RuntimeError(
                "no API keys: set GEMINI_KEYS_FILE (one key per line) or GEMINI_API_KEY"
            )
        self._keys = keys
        self._lock = threading.Lock()
        self._next_free = {k: 0.0 for k in keys}   # unix time each key is usable
        self._rr = 0

    def acquire(self) -> str:
        """Block until some key is available; return it and reserve its slot."""
        while True:
            with self._lock:
                now = time.time()
                order = self._keys[self._rr:] + self._keys[:self._rr]
                best = min(order, key=lambda k: self._next_free[k])
                wait = self._next_free[best] - now
                if wait <= 0:
                    self._next_free[best] = now + PER_KEY_MIN_INTERVAL
                    self._rr = (self._keys.index(best) + 1) % len(self._keys)
                    return best
            time.sleep(min(max(wait, 0.05), 2.0))

    def cooldown(self, key: str, seconds: float = QUOTA_COOLDOWN):
        with self._lock:
            self._next_free[key] = time.time() + seconds


_pool: _KeyPool | None = None
_pool_lock = threading.Lock()


def _get_pool() -> _KeyPool:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = _KeyPool()
        return _pool


def _is_quota_error(e: Exception) -> bool:
    s = f"{type(e).__name__}: {e}".lower()
    return ("429" in s or "resource_exhausted" in s or "resourceexhausted" in s
            or "quota" in s or "rate limit" in s)


def _is_retryable(e: Exception) -> bool:
    """Timeouts / transient transport errors: retry on another key slot."""
    s = f"{type(e).__name__}: {e}".lower()
    return ("timeout" in s or "timed out" in s or "deadline" in s
            or "503" in s or "500 " in s or "connection" in s
            or "unavailable" in s)


def generate(
    prompt: str,
    system_instruction: str = None,
    response_schema: dict = None,
    model: str = DEFAULT_MODEL,
    thinking_level: str = "HIGH",
    temperature: float = None,
    max_key_attempts: int = 10,
):
    """Call the API with structured JSON output. Returns the parsed dict.

    Rotates across the key pool; on quota errors the throttled key cools down
    and the call retries with the next key.
    """
    pool = _get_pool()
    last_err = None
    for _ in range(max_key_attempts):
        key = pool.acquire()
        try:
            client = genai.Client(
                api_key=key,
                http_options=types.HttpOptions(timeout=180_000),  # 180s per call
            )

            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                ),
            ]
            tools = [types.Tool(googleSearch=types.GoogleSearch())]
            config_kwargs = {
                "thinking_config": types.ThinkingConfig(thinking_level=thinking_level),
                "tools": tools,
                "response_mime_type": "application/json",
            }
            if response_schema:
                config_kwargs["response_schema"] = response_schema
            if system_instruction:
                config_kwargs["system_instruction"] = [
                    types.Part.from_text(text=system_instruction)
                ]
            if temperature is not None:
                config_kwargs["temperature"] = temperature

            generate_content_config = types.GenerateContentConfig(**config_kwargs)

            chunks = []
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text:
                    chunks.append(chunk.text)
            return json.loads("".join(chunks))
        except Exception as e:
            last_err = e
            if _is_quota_error(e):
                pool.cooldown(key)
                continue
            # Truncated/garbled model JSON (token-capped stream) is transient:
            # retrying on a fresh slot usually succeeds; a hard raise here
            # permanently discards the candidate that triggered it.
            if _is_retryable(e) or isinstance(e, json.JSONDecodeError):
                continue
            raise
    raise RuntimeError(f"all key attempts exhausted; last error: {last_err}")
