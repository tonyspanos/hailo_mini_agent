"""
LLM provider: hailo-ollama on a Hailo-10H (Raspberry Pi AI HAT+ 2).

hailo-ollama (Hailo Model Zoo GenAI, HailoRT 5.4.0+) serves an Ollama-compatible
REST API, default http://localhost:8000. This module talks to POST /api/chat
with the standard library only, so the agent has no SDK dependency.

Tool calling, two modes:
  native (default) — send Ollama's `tools` field; the server returns
                     message.tool_calls (arguments as a dict). Verified on
                     5.4.0 with qwen3:1.7b; documented officially from 5.5.0.
  prompt           — describe the tools in the system prompt and parse a JSON
                     reply into a ToolCall. For servers or models without
                     tool support (e.g. gemma4:e2b).
In native mode, an HTTP 400 "does not support tools" switches the provider to
prompt mode automatically for the rest of the session.

Interface: .complete(messages, tools) → LLMMessage. Messages are canonical
(OpenAI-style) dicts; this module translates them to the Ollama chat format.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass


DEFAULT_HOST = "http://localhost:8000"
DEFAULT_MODEL = "qwen3:1.7b"


# ---------------------------------------------------------------------------
# Canonical message types
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict  # already parsed from JSON


@dataclass
class LLMMessage:
    role: str
    content: str | None
    tool_calls: list[ToolCall] | None = None

    def to_dict(self) -> dict:
        """Canonical dict for the agent's messages list."""
        d: dict = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in self.tool_calls
            ]
        return d


class LLMProvider:
    name: str = "base"

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMMessage:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# Hailo-Ollama provider
# ---------------------------------------------------------------------------

_TOOL_PROMPT = """\
You can call tools. Available tools:
{tool_list}

To call a tool, reply with ONLY one JSON object on a single line and nothing else:
{{"tool": "<tool name>", "arguments": {{<arguments>}}}}

After a tool call you will receive the result in a message starting with "TOOL RESULT".
When you have enough information, reply with the final answer in plain text (no JSON)."""

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class HailoOllamaProvider(LLMProvider):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        tool_mode: str = "native",
        timeout: float = 300.0,
    ):
        if tool_mode not in ("native", "prompt"):
            raise ValueError("tool_mode must be 'native' or 'prompt'")
        self.model = model
        self.host = _normalize_host(host)
        self.tool_mode = tool_mode
        self.timeout = timeout
        self.name = f"hailo/{model}"
        self._call_counter = 0

    # -- HTTP ---------------------------------------------------------------

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.host + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:300]
            raise HailoHTTPError(e.code, path, detail) from e
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot reach hailo-ollama at {self.host} ({e.reason}). "
                f"Is `hailo-ollama` running? Set HAILO_OLLAMA_HOST if it is elsewhere."
            ) from e

        # stream=False gives one JSON object; tolerate NDJSON chunks just in case.
        lines = [ln for ln in body.splitlines() if ln.strip()]
        if len(lines) <= 1:
            return json.loads(body)
        chunks = [json.loads(ln) for ln in lines]
        merged = dict(chunks[-1])
        merged["message"] = {
            "role": "assistant",
            "content": "".join(c.get("message", {}).get("content", "") for c in chunks),
            "tool_calls": [tc for c in chunks for tc in c.get("message", {}).get("tool_calls") or []],
        }
        return merged

    # -- LLMProvider --------------------------------------------------------

    def complete(self, messages: list[dict], tools: list[dict]) -> LLMMessage:
        try:
            data = self._chat(messages, tools)
        except HailoHTTPError as e:
            if self.tool_mode == "native" and tools and e.status == 400 and "support tools" in e.detail:
                self.tool_mode = "prompt"
                data = self._chat(messages, tools)
            else:
                raise

        msg = data.get("message", {})
        content = _THINK_RE.sub("", msg.get("content") or "").strip()

        tool_calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args or "{}")
            tool_calls.append(ToolCall(id=self._next_id(), name=fn.get("name", ""), arguments=args))

        # Small models sometimes print the call as text even in native mode.
        if not tool_calls and tools:
            parsed = _parse_text_tool_call(content, {t["function"]["name"] for t in tools})
            if parsed:
                name, args = parsed
                tool_calls.append(ToolCall(id=self._next_id(), name=name, arguments=args))
                content = ""

        return LLMMessage(role="assistant", content=content or None, tool_calls=tool_calls or None)

    def _chat(self, messages: list[dict], tools: list[dict]) -> dict:
        native = self.tool_mode == "native"
        payload: dict = {
            "model": self.model,
            "messages": _to_ollama_messages(messages, tools, native),
            "stream": False,
        }
        if tools and native:
            payload["tools"] = tools
        return self._post("/api/chat", payload)

    def _next_id(self) -> str:
        self._call_counter += 1
        return f"hailo_call_{self._call_counter}"


class HailoHTTPError(RuntimeError):
    def __init__(self, status: int, path: str, detail: str):
        super().__init__(f"hailo-ollama HTTP {status} on {path}: {detail}")
        self.status = status
        self.detail = detail


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------

def _to_ollama_messages(canonical: list[dict], tools: list[dict], native: bool) -> list[dict]:
    """Canonical → Ollama /api/chat messages. In prompt mode, tools live in the system prompt."""
    if tools and not native:
        tool_list = "\n".join(
            f"- {t['function']['name']}: {t['function']['description']} "
            f"Arguments (JSON schema): {json.dumps(t['function']['parameters'].get('properties', {}))}"
            for t in tools
        )
        tool_prompt = _TOOL_PROMPT.format(tool_list=tool_list)
        if canonical and canonical[0]["role"] == "system":
            canonical = [{"role": "system", "content": canonical[0]["content"] + "\n\n" + tool_prompt}] + canonical[1:]
        else:
            canonical = [{"role": "system", "content": tool_prompt}] + canonical

    result: list[dict] = []
    tool_names: dict[str, str] = {}  # tool_call_id → tool name

    for msg in canonical:
        role = msg["role"]
        if role == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tool_names[tc["id"]] = tc["name"]
            if native:
                result.append({
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "tool_calls": [
                        {"function": {"name": tc["name"], "arguments": tc["arguments"]}}
                        for tc in msg["tool_calls"]
                    ],
                })
            else:
                # Replay the call in the JSON shape the prompt asked for.
                result.append({
                    "role": "assistant",
                    "content": "\n".join(
                        json.dumps({"tool": tc["name"], "arguments": tc["arguments"]})
                        for tc in msg["tool_calls"]
                    ),
                })
        elif role == "tool":
            name = tool_names.get(msg.get("tool_call_id", ""), "tool")
            if native:
                result.append({"role": "tool", "tool_name": name, "content": msg["content"]})
            else:
                result.append({"role": "user", "content": f"TOOL RESULT ({name}): {msg['content']}"})
        else:
            result.append({"role": role, "content": msg.get("content") or ""})
    return result


def _parse_text_tool_call(text: str, known: set[str]) -> tuple[str, dict] | None:
    """Find a {"tool": ..., "arguments": {...}} object in model text. Returns (name, args) or None."""
    if not text:
        return None
    decoder = json.JSONDecoder()
    i = text.find("{")
    while i != -1:
        try:
            obj, _ = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            name = obj.get("tool") or obj.get("name") or obj.get("function")
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if isinstance(name, str) and name in known and isinstance(args, dict):
                return name, args
        i = text.find("{", i + 1)
    return None


def _normalize_host(host: str) -> str:
    """Accept 'http://h:p', 'h:p' or ':p' (OLLAMA_HOST style)."""
    host = host.strip().rstrip("/")
    if not host.startswith(("http://", "https://")):
        if host.startswith(":"):
            host = "localhost" + host
        host = host.replace("0.0.0.0", "localhost")
        host = "http://" + host
    return host


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_provider(model: str | None = None) -> LLMProvider:
    """
    Create the hailo-ollama provider.

    Env: HAILO_OLLAMA_HOST (default http://localhost:8000; OLLAMA_HOST is used
    if HAILO_OLLAMA_HOST is unset), HAILO_TOOL_MODE (native | prompt).
    """
    host = os.environ.get("HAILO_OLLAMA_HOST") or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST
    return HailoOllamaProvider(
        model=model or DEFAULT_MODEL,
        host=host,
        tool_mode=os.environ.get("HAILO_TOOL_MODE", "native"),
    )
