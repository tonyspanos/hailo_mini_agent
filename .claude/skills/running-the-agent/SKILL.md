---
name: running-the-agent
description: >-
  How to run mini_agent on a Hailo-10H through hailo-ollama, including models,
  env vars, tool-calling modes, MCP, interactive REPL, and model fallback. Use
  when running agent.py, changing providers.py, debugging tool calling, or
  setting up .env.
---

# Running the agent

```bash
# On the Pi: hailo-ollama (HailoRT 5.4.0+) on :8000. Dev box here: HAILO_OLLAMA_HOST=http://pi5-ai:8000
hailo-ollama
curl http://localhost:8000/hailo/v1/list
curl http://localhost:8000/api/pull -H 'Content-Type: application/json' -d '{"model": "qwen3:1.7b", "stream": true}'

source .venv/bin/activate
python agent.py "What is 15% of 847?"
python agent.py --model qwen3:4b "..."            # 8 GB Hailo-10H only
python agent.py --fallback qwen3:4b qwen2.5:1.5b "..."
python agent.py --interactive
python agent.py --quiet "What is 144 * 37?"
```

## File roles

| File | Role |
|---|---|
| `agent.py` | CLI: argparse, REPL |
| `core.py` | `run_agent` |
| `reliability.py` | retry, circuit breaker, validation, tracing, fallback |
| `providers.py` | `HailoOllamaProvider`, `create_provider(model)` |
| `tools.py` | tools, schemas, `call_tool` |
| `ui.py` | spinner |
| `mcp_server.py` / `mcp_client.py` | MCP demo tools |

hailo-ollama is the only provider. It is called on `POST /api/chat` with stdlib `urllib`, `stream: false`. The canonical messages list is OpenAI-style; `_to_ollama_messages` converts it.

## Tool calling

- `HAILO_TOOL_MODE=native` (default): sends Ollama's `tools` field. Verified on 5.4.0 + `qwen3:1.7b`: `tool_calls[].function.arguments` comes back as a dict. Tool results go back as `role: tool` with `tool_name`.
- `HAILO_TOOL_MODE=prompt`: tools are described in the system prompt, the model replies `{"tool": ..., "arguments": {...}}`, and results go back as `TOOL RESULT (name): ...` user messages.
- An HTTP 400 "does not support tools" in native mode switches to prompt mode automatically.
- `<think>…</think>` blocks (qwen3) are stripped from content.

## Env

| Var | Default |
|---|---|
| `HAILO_OLLAMA_HOST` | `http://localhost:8000` (falls back to `OLLAMA_HOST`) |
| `HAILO_TOOL_MODE` | `native` |

No API keys.

## Models

Default `qwen3:1.7b` (about 15-30 s per agent turn on the Pi; reliable for one tool per task, may skip a second). `qwen3:4b` (5.5.0+) needs the 8 GB device. `qwen2.5:1.5b` and `qwen2:1.5b` are smaller fallbacks. `gemma4:e2b` has no tool support, so it runs in prompt mode. The server runs one model and one connection at a time.

## Adding tools

1. Function in `tools.py`
2. `TOOL_FUNCTIONS`
3. OpenAI-format schema in `TOOL_SCHEMAS`

`web_search` and `get_weather` are stubs.

## Tests

`python -m unittest test_hailo_provider test_tools` runs offline against a fake `/api/chat` server.
