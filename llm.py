"""LLM client for the self-hosted Qwen wrapper (FastAPI in front of vLLM).

This replaces the Anthropic SDK. The wrapper exposes:

    POST /chat/text   header: api-key
    body: {"prompt": str, "max_tokens": int, "temperature": float}
    -> {"response": "<model text>"}

It uses a fixed system prompt and has no tool-use / structured-output mode, so
`structured()` folds the caller's system prompt + JSON schema into the single
`prompt` field, asks the model for JSON only, and parses it back out of the
reply (with one retry). Credentials/URL come from the environment (.env):

    QWEN_API_URL   base URL of the wrapper   (default http://127.0.0.1:8000)
    QWEN_API_KEY   the api-key the wrapper expects
    QWEN_TIMEOUT   per-request timeout seconds (default 300 — vision models are slow)
"""

from __future__ import annotations

import json
import os
import re

import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("QWEN_API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("QWEN_API_KEY", "")
TIMEOUT = float(os.getenv("QWEN_TIMEOUT", "300"))

_ENDPOINT_SUFFIXES = ("/chat/text", "/chat/vision", "/chat/compare")


def _base_url(url: str) -> str:
    """Accept either the wrapper base URL or a full /chat/* endpoint URL, and
    return the base (so we can safely append /chat/text ourselves)."""
    url = url.strip().rstrip("/")
    for suffix in _ENDPOINT_SUFFIXES:
        if url.endswith(suffix):
            return url[: -len(suffix)].rstrip("/")
    return url

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _extract_json(text: str) -> dict:
    """Pull the first balanced JSON object out of a free-form model reply.

    Tolerates ```json fences and leading/trailing prose ("Here is the JSON: ...").
    """
    s = _FENCE.sub("", text.strip()).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    start = s.find("{")
    if start == -1:
        raise ValueError("no JSON object in model reply")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(s[start:i + 1])
    raise ValueError("unbalanced JSON in model reply")


class QwenClient:
    """Thin sync client over the Qwen wrapper's /chat/text endpoint."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 timeout: float | None = None):
        self.base_url = _base_url(base_url or API_URL)
        self.api_key = api_key if api_key is not None else API_KEY
        self.timeout = timeout or TIMEOUT

    def chat_text(self, prompt: str, max_tokens: int,
                  temperature: float = 0.2) -> str:
        r = httpx.post(
            f"{self.base_url}/chat/text",
            headers={"api-key": self.api_key},
            json={"prompt": prompt, "max_tokens": max_tokens,
                  "temperature": temperature},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["response"]

    def structured(self, system: str, user: str, schema: dict, max_tokens: int,
                   temperature: float = 0.2, retries: int = 1) -> tuple[dict, str]:
        """Get a schema-conforming JSON object back from a text-only endpoint.

        Returns (parsed_dict, raw_text). Raw text is handed back so the caller
        can estimate output tokens (the wrapper reports no usage)."""
        instructions = (
            "Respond with ONLY a single JSON object that conforms to this JSON "
            "schema. Do not include any explanation, preamble, or markdown code "
            f"fences.\n\nJSON schema:\n{json.dumps(schema)}"
        )
        prompt = f"{system}\n\n{user}\n\n{instructions}"
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            text = self.chat_text(prompt, max_tokens, temperature)
            try:
                return _extract_json(text), text
            except (ValueError, json.JSONDecodeError) as e:
                last_err = e
                prompt = (
                    f"{system}\n\n{user}\n\n{instructions}\n\n"
                    "Your previous reply was not valid JSON. Return ONLY the "
                    "JSON object, nothing else."
                )
        raise RuntimeError(f"model did not return valid JSON: {last_err}")
