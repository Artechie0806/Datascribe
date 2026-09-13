"""LLM client for an OpenAI-compatible server, tuned for small local models.

Talks to anything that speaks `POST /v1/chat/completions` — LM Studio,
llama.cpp, Ollama, vLLM — and is written around the three ways a 7-14B model
fails at structured output where a frontier model does not:

1. It thinks the budget away. Qwen3-class models emit a reasoning block before
   the answer. Ask for 700 tokens and the reply is `finish_reason: "length"`
   with an EMPTY `content` — every token went to `reasoning_content`. That is
   the "no JSON object in model reply" error: there was no reply at all.
   Fixed by sending `reasoning_effort` (default "none"), by floor-ing the
   generation cap well above the context reserve, and by retrying a truncated
   call with a bigger cap instead of re-asking at the same size.

2. It writes JSON by hand, badly. Raw newlines inside a string (a multi-line
   SELECT), trailing commas, ```json fences, "Here is the JSON:" preambles,
   Python True/None. Fixed by asking the server to constrain decoding to the
   schema (`response_format: json_schema`) so the grammar makes invalid JSON
   unrepresentable — and, when the server or model cannot, by repairing and
   then salvaging the reply rather than throwing it away.

3. Constrained decoding makes it dumber. With a grammar forcing the first token
   to be `{"sql":`, the model has to answer before it has thought, and the SQL
   gets measurably worse. Fixed in the schemas themselves: the rationale field
   is declared FIRST, so the model writes its reasoning into the JSON and
   answers after it. (See the field order in agents.py.)

Capabilities are probed by use, not asserted: if a server rejects
`response_format`, `reasoning_effort` or `model`, that feature is switched off
for the rest of the process and the call is retried. A plain OpenAI-compatible
endpoint with none of them still works, just with a longer retry ladder.

Environment (.env) — QWEN_* names still work:

    LLM_API_URL           base URL, with or without /v1   (default http://127.0.0.1:1234)
    LLM_API_KEY           bearer token; local servers ignore it
    LLM_MODEL             model id; auto-resolved from /v1/models when unset
    LLM_TIMEOUT           per-request seconds (default 600 — local models are slow)
    LLM_CONTEXT           the server's context window (default 16000)
    LLM_REASONING_EFFORT  none | low | medium | high (default none)
    LLM_MIN_OUTPUT        floor for max_tokens        (default 1024)
    LLM_MAX_OUTPUT        ceiling for max_tokens      (default 4096)
    LLM_THINK_HEADROOM    extra tokens when reasoning is on (default 2048)
    LLM_JSON_SCHEMA       1 | 0 — force constrained decoding on/off
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field

import httpx
from dotenv import load_dotenv

load_dotenv()


def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.getenv(n)
        if v is not None and v.strip():
            return v.strip()
    return default


def _env_int(*names: str, default: int) -> int:
    try:
        return int(float(_env(*names, default=str(default))))
    except ValueError:
        return default


API_URL = _env("LLM_API_URL", "QWEN_API_URL", default="http://127.0.0.1:1234")
API_KEY = _env("LLM_API_KEY", "QWEN_API_KEY", default="")
TIMEOUT = float(_env("LLM_TIMEOUT", "QWEN_TIMEOUT", default="600"))
MODEL = _env("LLM_MODEL", default="")

# What the server can hold. Everything in context.py sizes prompts against this.
CONTEXT_WINDOW = _env_int("LLM_CONTEXT", "QWEN_CONTEXT", default=16000)

# Reasoning models bill their thinking to the same budget as the answer, so the
# generation cap has to be larger than the space the answer needs.
REASONING_EFFORT = _env("LLM_REASONING_EFFORT", default="none").lower()
MIN_OUTPUT = _env_int("LLM_MIN_OUTPUT", default=1024)
MAX_OUTPUT = _env_int("LLM_MAX_OUTPUT", default=4096)
THINK_HEADROOM = _env_int("LLM_THINK_HEADROOM", default=2048)

_FORCE_JSON_SCHEMA = _env("LLM_JSON_SCHEMA", default="")

# Efforts that mean "do not think". Anything else buys reasoning headroom.
_NO_THINK = {"none", "off", "minimal", "0", "disable", "disabled"}

_ENDPOINT_SUFFIXES = ("/chat/text", "/chat/vision", "/chat/compare",
                      "/chat/completions", "/completions")


def _base_url(url: str) -> str:
    """Accept a base URL or a full endpoint URL and return the /v1 root."""
    url = (url or "").strip().rstrip("/")
    for suffix in _ENDPOINT_SUFFIXES:
        if url.endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")
            break
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


# --- server capabilities ----------------------------------------------------
@dataclass
class Caps:
    """What this server turned out to support. Probed by use: a 400 naming a
    field switches that field off for good rather than failing the call.

    Shared per base URL, because the app builds a fresh client per request and
    re-discovering this on every turn would cost a wasted round trip each time.
    """
    json_schema: bool | None = None      # None = not tried yet
    reasoning_effort: bool | None = None
    model: str | None = None
    model_resolved: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


_CAPS: dict[str, Caps] = {}
_CAPS_LOCK = threading.Lock()


def _caps_for(base_url: str) -> Caps:
    with _CAPS_LOCK:
        return _CAPS.setdefault(base_url, Caps())


def reset_caps() -> None:
    """Forget what we learned about every server (tests, config reloads)."""
    with _CAPS_LOCK:
        _CAPS.clear()


# --- prompt-side schema rendering ------------------------------------------
def _type_hint(spec: dict) -> str:
    t = spec.get("type", "string")
    if "enum" in spec:
        return "one of: " + ", ".join(str(e) for e in spec["enum"])
    if t == "array":
        inner = spec.get("items", {})
        return f"array of {_type_hint(inner) if inner else 'string'}"
    if t == "object":
        keys = ", ".join(spec.get("properties", {}))
        return f"object with {keys}" if keys else "object"
    return {"string": "string", "boolean": "true or false",
            "number": "number", "integer": "whole number"}.get(t, str(t))


def describe_schema(schema: dict) -> str:
    """The schema as a field list rather than a JSON Schema dump.

    A 9B model reads `mode: one of: chat, data` far more reliably than it reads
    a nested `{"type":"object","properties":{...}}`, and it costs a third of the
    tokens."""
    props = schema.get("properties") or {}
    if not props:
        return "a JSON object"
    return "\n".join(f"  {name}: {_type_hint(spec)}" for name, spec in props.items())


def _example(schema: dict) -> str:
    """A filled-in skeleton, for the unconstrained fallback only — a model that
    has already failed to produce JSON needs to be shown the shape, not told."""
    def value(spec: dict):
        t = spec.get("type", "string")
        if "enum" in spec:
            return spec["enum"][0]
        if t == "array":
            return [value(spec.get("items", {"type": "string"}))]
        if t == "object":
            return {k: value(v) for k, v in (spec.get("properties") or {}).items()}
        if t == "boolean":
            return True
        if t in ("number", "integer"):
            return 0
        return "..."
    return json.dumps(value(schema))


def _strictify(schema: dict) -> dict:
    """Make a schema legal for strict constrained decoding.

    Grammar backends want every object closed (`additionalProperties: false`)
    and, under OpenAI's `strict`, every property required. Forcing `required`
    is not just protocol compliance — left optional, a small model emits the
    one field it is sure about and drops the rest, so the router returns
    `{"mode": "data"}` and silently throws away the rewritten question. Every
    caller reads fields with a default, so a forced-but-empty field is free."""
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    if out.get("type") == "object" or "properties" in out:
        props = {k: _strictify(v) for k, v in (out.get("properties") or {}).items()}
        out["properties"] = props
        out["required"] = list(props)
        out["additionalProperties"] = False
    if "items" in out:
        out["items"] = _strictify(out["items"])
    return out


# --- reading a reply apart --------------------------------------------------
# Reasoning left inline in `content` by servers that do not split it out.
_THINK_BLOCK = re.compile(
    r"<(think|thinking|reasoning|analysis)\b[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<(think|thinking|reasoning|analysis)\b[^>]*>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</(think|thinking|reasoning|analysis)>", re.IGNORECASE)
_FENCE_BLOCK = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n?(.*?)```", re.DOTALL)
_FENCE_EDGE = re.compile(r"^```[a-zA-Z0-9_+-]*\s*|\s*```$", re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Drop inline thinking. A dangling `</think>` with no opener means the
    block was truncated mid-thought — everything before it is reasoning."""
    if not text:
        return ""
    s = _THINK_BLOCK.sub("", text)
    close = None
    for close in _THINK_CLOSE.finditer(s):
        pass
    if close is not None:
        s = s[close.end():]
    open_ = _THINK_OPEN.search(s)
    if open_ is not None:          # opened and never closed: nothing usable after
        s = s[:open_.start()]
    return s.strip()


def _candidates(text: str) -> list[str]:
    """Every balanced {...} in the reply, outermost first, in order.

    Not just the first: a model that narrates before answering ("the schema is
    {...} so:") puts a decoy object ahead of the real one, and a model that
    fences its answer puts the real one last."""
    out: list[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(text[start:i + 1])
    if depth > 0 and start >= 0:   # truncated mid-object: keep it for repair
        out.append(text[start:])
    return out


_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_SMART = str.maketrans({"“": '"', "”": '"', "‘": "'",
                        "’": "'", " ": " "})
_PY_LITERAL = re.compile(r"(?<![\"\w])(True|False|None)(?![\"\w])")


def _escape_controls(s: str) -> str:
    """Escape raw newlines/tabs that appear INSIDE a JSON string.

    This is the single most common way a hand-written reply is invalid: the SQL
    author writes a formatted multi-line SELECT straight into `"sql": "..."`."""
    out: list[str] = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
                out.append(ch)
                continue
            if ch == "\\":
                esc = True
                out.append(ch)
                continue
            if ch == '"':
                in_str = False
                out.append(ch)
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            if ord(ch) < 0x20:
                out.append(f"\\u{ord(ch):04x}")
                continue
            out.append(ch)
            continue
        if ch == '"':
            in_str = True
        out.append(ch)
    return "".join(out)


def _close_truncated(s: str) -> str:
    """Shut an object the model ran out of tokens mid-way through.

    A truncated reply usually holds every field that matters and dies inside
    the last one; closing the string and the brackets recovers the rest."""
    depth_stack: list[str] = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth_stack.append(ch)
        elif ch in "}]":
            if depth_stack:
                depth_stack.pop()
    if not in_str and not depth_stack:
        return s
    out = s
    if in_str:
        out = out.rstrip("\\") + '"'
    out = _TRAILING_COMMA.sub(r"\1", out.rstrip().rstrip(","))
    for opener in reversed(depth_stack):
        out += "}" if opener == "{" else "]"
    return out


def _repair(s: str) -> str:
    s = s.translate(_SMART)
    s = _escape_controls(s)
    s = _TRAILING_COMMA.sub(r"\1", s)
    s = _PY_LITERAL.sub(lambda m: {"True": "true", "False": "false",
                                   "None": "null"}[m.group(1)], s)
    return _close_truncated(s)


def _score(obj: dict, schema: dict) -> int:
    required = schema.get("required") or list((schema.get("properties") or {}))
    return sum(1 for k in required if k in obj)


def extract_json(text: str, schema: dict | None = None) -> dict:
    """Pull the intended JSON object out of a free-form reply.

    Tolerates reasoning blocks, ```json fences, prose on either side, a decoy
    object before the real one, and — via `_repair` — unescaped newlines,
    trailing commas, Python literals and an object the model was cut off in the
    middle of. When several objects parse, the one carrying the most of the
    schema's fields wins."""
    schema = schema or {}
    body = strip_reasoning(text)
    if not body.strip():
        raise ValueError("model returned an empty reply")

    pool = [body]
    pool += [m.group(1) for m in _FENCE_BLOCK.finditer(body)]
    pool.append(_FENCE_EDGE.sub("", body.strip()))
    for chunk in list(pool):
        pool.extend(_candidates(chunk))

    best: dict | None = None
    best_score = -1
    seen: set[str] = set()
    for raw in pool:
        raw = raw.strip()
        if not raw.startswith("{") or raw in seen:
            continue
        seen.add(raw)
        for variant in (raw, _repair(raw)):
            try:
                obj = json.loads(variant)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(obj, dict):
                continue
            s = _score(obj, schema)
            if s > best_score:
                best, best_score = obj, s
            break
    if best is not None:
        return best
    raise ValueError("no JSON object in model reply")


# --- last resort: read the fields out of prose ------------------------------
def _string_at(text: str, key: str) -> str | None:
    m = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"', text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(f'"{_escape_controls(m.group(1))}"')
    except (json.JSONDecodeError, ValueError):
        return m.group(1)


def salvage(text: str, schema: dict) -> dict:
    """Recover what we can from a reply that will not parse as an object.

    Two shapes show up constantly with small models and both are recoverable:
    a mangled object whose individual `"key": "value"` pairs are still intact,
    and an answer given as bare prose or a bare ```sql fence when the model gave
    up on the wrapper entirely. Raises if nothing required comes back, so a
    genuinely empty reply still fails loudly."""
    props = schema.get("properties") or {}
    required = schema.get("required") or list(props)
    body = strip_reasoning(text)
    out: dict = {}

    for key, spec in props.items():
        t = spec.get("type", "string")
        if t == "string":
            v = _string_at(body, key)
            if v is not None:
                out[key] = v
        elif t == "boolean":
            m = re.search(rf'"{re.escape(key)}"\s*:\s*(true|false|True|False)', body)
            if m:
                out[key] = m.group(1).lower() == "true"
        elif t in ("number", "integer"):
            m = re.search(rf'"{re.escape(key)}"\s*:\s*(-?\d+(?:\.\d+)?)', body)
            if m:
                out[key] = float(m.group(1)) if t == "number" else int(float(m.group(1)))
        elif t == "array":
            m = re.search(rf'"{re.escape(key)}"\s*:\s*\[(.*?)\]', body, re.DOTALL)
            if m:
                out[key] = re.findall(r'"((?:\\.|[^"\\])*)"', m.group(1))

    missing = [k for k in required if k not in out]
    # A bare answer: the model wrote the SQL, or the sentence, with no wrapper.
    if missing and len(missing) == 1:
        key = missing[0]
        if (props.get(key, {}).get("type", "string")) == "string":
            fence = _FENCE_BLOCK.search(body)
            bare = (fence.group(1) if fence else body).strip()
            if bare and not bare.lstrip().startswith("{"):
                out[key] = bare
                missing = []

    if missing:
        raise ValueError(f"could not recover {', '.join(missing)} from the reply")
    return out


# --- the client -------------------------------------------------------------
@dataclass
class Reply:
    text: str = ""
    reasoning: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"

    @property
    def thought_it_away(self) -> bool:
        """Truncated with nothing to show for it: the whole budget went to
        reasoning. Retrying at the same size just burns it again."""
        return self.truncated and not self.text.strip()


class QwenClient:
    """Sync client for an OpenAI-compatible server (LM Studio, llama.cpp, vLLM).

    The name is historical; nothing here is Qwen-specific."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ):
        self.base_url = _base_url(base_url or API_URL)
        self.api_key = api_key or API_KEY or "local"
        self.timeout = timeout or TIMEOUT
        self.reasoning_effort = (reasoning_effort or REASONING_EFFORT).lower()
        self.caps = _caps_for(self.base_url)
        if model or MODEL:
            self.caps.model = model or MODEL
            self.caps.model_resolved = True

    # --- model discovery ----------------------------------------------------
    def _model(self) -> str | None:
        """Name the model explicitly when we can. LM Studio will serve whatever
        is loaded, but llama.cpp and vLLM both reject an unknown id and some
        setups have an embedding model loaded alongside the chat one."""
        caps = self.caps
        if caps.model_resolved:
            return caps.model
        with caps.lock:
            if caps.model_resolved:
                return caps.model
            try:
                r = httpx.get(f"{self.base_url}/models",
                              headers=self._headers(), timeout=15)
                r.raise_for_status()
                ids = [m.get("id", "") for m in r.json().get("data", [])]
                chat = [i for i in ids if i and not re.search(
                    r"embed|rerank|whisper|clip", i, re.IGNORECASE)]
                caps.model = chat[0] if chat else (ids[0] if ids else None)
            except Exception:
                caps.model = None
            caps.model_resolved = True
            return caps.model

    def health(self) -> dict:
        """What the app is actually pointed at — surfaced on the dataset page so
        a misconfigured URL is visible before the first question, not after."""
        try:
            r = httpx.get(f"{self.base_url}/models", headers=self._headers(),
                          timeout=10)
            r.raise_for_status()
            return {"ok": True, "url": self.base_url,
                    "model": self._model(),
                    "models": [m.get("id") for m in r.json().get("data", [])]}
        except Exception as e:
            return {"ok": False, "url": self.base_url, "error": str(e)}

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    # --- budgeting ----------------------------------------------------------
    def _output_budget(self, reserve: int, multiplier: int = 1) -> int:
        """Turn a context reserve into a generation cap.

        These are not the same number. The caller's reserve is how much room the
        ANSWER needs; `max_tokens` also has to cover whatever the model thinks
        first, and a reasoning model will happily spend 900 tokens deciding a
        two-field routing call. Floor it, add headroom when thinking is on, and
        cap it so a runaway loop cannot eat the window."""
        n = max(int(reserve or 0), MIN_OUTPUT)
        if self.reasoning_effort not in _NO_THINK:
            n += THINK_HEADROOM
        return max(MIN_OUTPUT, min(n * multiplier, MAX_OUTPUT * multiplier))

    # --- one request, degrading on unsupported fields ----------------------
    def _complete(self, messages: list[dict], max_tokens: int, temperature: float,
                  schema: dict | None = None,
                  reasoning_effort: str | None = None) -> Reply:
        caps = self.caps
        effort = reasoning_effort if reasoning_effort is not None \
            else self.reasoning_effort

        want_schema = schema is not None and _FORCE_JSON_SCHEMA != "0" \
            and caps.json_schema is not False
        if _FORCE_JSON_SCHEMA == "1" and schema is not None:
            want_schema = True

        for _ in range(4):
            body: dict = {"messages": messages, "max_tokens": max_tokens,
                          "temperature": temperature, "stream": False}
            model = self._model()
            if model:
                body["model"] = model
            if want_schema:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "response", "strict": True,
                                    "schema": _strictify(schema)},
                }
            if effort and caps.reasoning_effort is not False:
                body["reasoning_effort"] = effort

            r = httpx.post(f"{self.base_url}/chat/completions",
                           headers=self._headers(), json=body,
                           timeout=self.timeout)

            if r.status_code in (400, 404, 422):
                note = r.text.lower()
                if want_schema and ("response_format" in note or "json_schema" in note
                                    or "grammar" in note or "schema" in note):
                    caps.json_schema = False
                    want_schema = False
                    continue
                if "reasoning" in note and caps.reasoning_effort is not False:
                    caps.reasoning_effort = False
                    continue
                if body.get("model") and "model" in note:
                    caps.model, caps.model_resolved = None, True
                    continue
            r.raise_for_status()

            if want_schema:
                caps.json_schema = True
            if "reasoning_effort" in body:
                caps.reasoning_effort = True
            return self._read(r.json())

        r.raise_for_status()
        return self._read(r.json())

    @staticmethod
    def _read(data: dict) -> Reply:
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        return Reply(
            text=msg.get("content") or "",
            reasoning=msg.get("reasoning_content") or msg.get("reasoning") or "",
            finish_reason=choice.get("finish_reason") or "",
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            reasoning_tokens=int(details.get("reasoning_tokens") or 0),
        )

    # --- public API ---------------------------------------------------------
    def chat_text(self, prompt: str, max_tokens: int = 2048,
                  temperature: float = 0.2, system: str = "") -> str:
        messages = ([{"role": "system", "content": system}] if system else []) \
            + [{"role": "user", "content": prompt}]
        reply = self._complete(messages, self._output_budget(max_tokens),
                               temperature)
        text = strip_reasoning(reply.text)
        if not text and reply.reasoning:
            text = reply.reasoning.strip()
        return text

    def structured(
        self,
        system: str,
        user: str,
        schema: dict,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        retries: int = 2,
    ) -> tuple[dict, str]:
        """Get a schema-conforming JSON object back from the model.

        Returns (object, raw reply text). The ladder escalates rather than
        repeating itself — re-asking a model that just thought its budget away,
        at the same budget, is how one failure becomes three:

            1. schema-constrained decoding at the caller's temperature
            2. truncated? same again with twice the room and thinking off.
               garbled? same again at temperature 0, shown its own bad reply
            3. unconstrained, with a worked example, and salvage the fields out
               of whatever comes back
        """
        base = [{"role": "system", "content": system},
                {"role": "user", "content":
                    f"{user}\n\nReturn a single JSON object with these fields:\n"
                    f"{describe_schema(schema)}"}]
        budget = self._output_budget(max_tokens)
        attempts = max(1, retries + 1)

        last_err: Exception | None = None
        last_reply = Reply()
        for attempt in range(attempts):
            final = attempt == attempts - 1
            messages = list(base)
            temp = temperature
            effort = None
            use_schema: dict | None = schema

            if attempt == 1:
                temp = 0.0
                if last_reply.truncated:
                    budget = self._output_budget(max_tokens, multiplier=2)
                    effort = "none"
                else:
                    messages.append({"role": "assistant",
                                     "content": (last_reply.text or "")[:600]})
                    messages.append({"role": "user", "content":
                                     "That was not a valid JSON object. Reply "
                                     "with the JSON object only — no prose, no "
                                     "code fences, no explanation outside it."})
            elif attempt >= 2:
                # Constrained decoding is not the problem any more; the wrapper
                # is. Drop the grammar, show the shape, and take what we get.
                temp = 0.0
                effort = "none"
                use_schema = None
                budget = self._output_budget(max_tokens, multiplier=2)
                messages[-1] = {"role": "user", "content":
                                f"{user}\n\nReply with ONLY a JSON object of "
                                f"this exact shape, on one line:\n"
                                f"{_example(schema)}\n\nFields:\n"
                                f"{describe_schema(schema)}"}

            try:
                reply = self._complete(messages, budget, temp, use_schema, effort)
            except httpx.HTTPError as e:
                last_err = e
                if final:
                    raise
                continue
            last_reply = reply

            raw = reply.text or reply.reasoning
            try:
                parsed = extract_json(reply.text, schema)
                # A cut-off reply can still parse: `_close_truncated` shuts the
                # brackets and hands back an object whose last field is half a
                # sentence — or half a SELECT, which then fails to EXPLAIN with
                # a parser error nobody can trace back to here. Only accept that
                # when there is no attempt left to spend on a bigger budget.
                if reply.truncated and not final:
                    last_err = ValueError(
                        f"reply truncated at {reply.completion_tokens} tokens")
                else:
                    return parsed, reply.text
            except (ValueError, json.JSONDecodeError) as e:
                last_err = e
            # Reasoning sometimes carries a complete draft the answer never got
            # to repeat, and on the last pass a half-answer beats no answer.
            if reply.reasoning:
                try:
                    return extract_json(reply.reasoning, schema), reply.text
                except (ValueError, json.JSONDecodeError):
                    pass
            if final:
                try:
                    return salvage(raw, schema), reply.text
                except ValueError as e:
                    last_err = e

        raise RuntimeError(
            f"model did not return valid JSON after {attempts} attempts: "
            f"{last_err}{self._diagnosis(last_reply)}")

    def _diagnosis(self, reply: Reply) -> str:
        """Say which failure this was. "no JSON object in model reply" is true
        of a model that wrote prose and of one that never got to write at all,
        and the fix is different."""
        if reply.thought_it_away:
            return (f" — the model spent its entire {reply.completion_tokens}-token "
                    f"budget on reasoning and returned no answer. Raise "
                    f"LLM_MAX_OUTPUT (now {MAX_OUTPUT}) or set "
                    f"LLM_REASONING_EFFORT=none.")
        if reply.truncated:
            return (f" — the reply was cut off at {reply.completion_tokens} tokens. "
                    f"Raise LLM_MAX_OUTPUT (now {MAX_OUTPUT}).")
        if self.caps.json_schema is False:
            return (" — this server rejected constrained decoding "
                    "(response_format: json_schema), so the model is writing "
                    "JSON by hand. A server that supports it is far more "
                    "reliable for small models.")
        head = (reply.text or "").strip().replace("\n", " ")[:200]
        return f" — the model replied: {head!r}" if head else ""
