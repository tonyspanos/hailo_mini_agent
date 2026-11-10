# Mini Agent

A minimal AI agent built without any agent framework — just Python, the standard library, and a `while` loop.

**This fork runs every LLM call on a Hailo-10H** (Raspberry Pi 5 + AI HAT+ 2) through [hailo-ollama](https://github.com/hailo-ai/hailo_model_zoo_genai), HailoRT 5.4.0+ (tested on 5.4.0). The cloud providers, plain Ollama, and mixed mode from upstream are removed.

This project accompanies the Medium article series by [Sergey Neskoromny](https://www.linkedin.com/in/sergey-neskoromny/):

## Software Engineering in the Agentic Era" series
- **Part 1:** [Building an AI Agent from Scratch: No Magic, Just a Deterministic Loop](https://levelup.gitconnected.com/building-an-ai-agent-from-scratch-no-magic-just-a-deterministic-loop-a916161705fb?sk=bd25f188cb4cb68d52ef5b77dd67336a)
- **Part 2:** [Your AI Agent Will Fail. Here's How to Make It Recoverable.](https://levelup.gitconnected.com/your-ai-agent-will-fail-heres-how-to-make-it-recoverable-781e0db1b5b3?sk=88eb3771ebb432ef27658457a546478f)
- **Part 3:** [Building Complete Systems from Day One: Why Simple-First Has Become Expensive in the AI Era](https://levelup.gitconnected.com/building-complete-systems-from-day-one-why-simple-first-has-become-expensive-in-the-ai-era-41dce4a708df?sk=1be72881afc072fb8f5d9bd724ecc60d)
- **Part 4:** [Give Your Testing Agent Eyes: A Visual Testing Agent from Scratch](https://levelup.gitconnected.com/give-your-testing-agent-eyes-a-visual-testing-agent-from-scratch-f52a63ce72ed?sk=e94ac212df1ed268e282f1daaa5c3c5e)
- **Part 5:** [The Agent Built the Feature in Four Minutes. Deploying It Took Me Forty.](https://levelup.gitconnected.com/the-agent-built-the-feature-in-four-minutes-deploying-it-took-me-forty-9eec9ba3c206?sk=5c4b2f529fb8ed6619f25e82bc412400)
- **Part 6:** [Your AI Agent Loads Two Instruction Files Every Turn. Keep Both Thin.](https://medium.com/gitconnected/your-ai-agent-loads-two-memory-files-every-turn-keep-both-thin-ec71c28f1bb0?sk=c279a1d565f8054d598c9f3704be0305)


Follow me on [LinkedIn](https://lnkd.in/epTFAmQJ) and [Medium](https://sergey-nes.medium.com/) for more on AI tools, mobile development, and whatever I'm currently building!

---

## What it does

The agent takes a task, calls tools when needed, observes the results, and loops until it has a final answer. The model runs on the Hailo-10H NPU via hailo-ollama; the tools run locally in Python. No API keys, no cloud.

## Setup

On the Raspberry Pi, install the [Hailo Model Zoo GenAI](https://github.com/hailo-ai/hailo_model_zoo_genai) package (HailoRT 5.4.0 or newer; 5.5.0 adds `qwen3:4b` and `gemma4:e2b`), then start the server and pull the model:

```bash
hailo-ollama                                   # listens on 0.0.0.0:8000 (OLLAMA_HOST to change)
curl http://localhost:8000/hailo/v1/list       # models your build supports
curl http://localhost:8000/api/pull -H 'Content-Type: application/json'      -d '{"model": "qwen3:1.7b", "stream": true}'
```

Then the agent:

```bash
git clone https://github.com/tonyspanos/hailo_mini_agent
cd hailo_mini_agent

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scriptsctivate

pip install -r requirements.txt
cp .env.example .env           # only needed if hailo-ollama is not on localhost:8000
```

| Env var | Default | Meaning |
|---|---|---|
| `HAILO_OLLAMA_HOST` | `http://localhost:8000` | Server address. Falls back to `OLLAMA_HOST` if unset. |
| `HAILO_TOOL_MODE` | `native` | `native`: Ollama `tools` field on `/api/chat` (works on 5.4.0 with `qwen3:1.7b`). `prompt`: tools described in the system prompt, JSON reply parsed. |

In `native` mode, if the server answers HTTP 400 "does not support tools" (e.g. `gemma4:e2b`), the provider switches to `prompt` mode by itself. `qwen3` replies may contain `<think>…</think>` blocks; the provider strips them.

hailo-ollama handles one connection and one model at a time, so do not run two agents against the same device.

---

## Models

| Model | Notes |
|---|---|
| `qwen3:1.7b` | **Default.** Fits the 4 GB and 8 GB Hailo-10H. Native tool calls verified on 5.4.0; it handles one tool per task well and may skip a second tool. |
| `qwen3:4b` | Qwen3-4B-Instruct-2507, new in 5.5.0. 8 GB Hailo-10H only. |
| `qwen2.5:1.5b`, `qwen2:1.5b`, `llama3.2:1b` | Older, smaller fallbacks (available on 5.4.0). |
| `gemma4:e2b` | New in 5.5.0. No tool calling through hailo-ollama; the agent uses prompt mode. |

Check the exact tags on your device with `curl http://localhost:8000/hailo/v1/list`.

---

## Usage

```bash
python agent.py "What is 15% of 847?"
python agent.py --model qwen3:4b "What's the weather in Tokyo and today's date?"
python agent.py --interactive
python agent.py --quiet "What is 144 * 37?"
```

### With model fallback

Try models in order; skip a model that errors or prints tool calls as text:

```bash
python agent.py --fallback qwen3:4b qwen2.5:1.5b "What is 15% of 847?"
```

Each attempt uses the full reliability stack from `reliability.py`.

### Resilience test (LLM retry)

Start a task, then stop `hailo-ollama` while you see `Thinking…`:

```
WARNING  LLM call attempt 1 failed (Cannot reach hailo-ollama …) — retrying in 2.1s
WARNING  LLM call attempt 2 failed (Cannot reach hailo-ollama …) — retrying in 4.3s
```

Restart it and the next attempt succeeds. The retry is `_RetryingProvider` in `reliability.py`: up to 5 attempts, base delay 2 s, doubling each retry.

### Multi-step example

```bash
python agent.py --model qwen3:4b "Get today's date and the weather in Tokyo. Calculate how many days are left until New Year's Day 2027. Write a short daily briefing with all three facts to briefing.txt, then read it back and count how many words it has."
```

This should trigger `get_current_date` → `get_weather` → `calculate` → `write_file` → `read_file` → `count_words`. Small models may skip steps; `qwen3:4b` follows long chains better than `qwen3:1.7b`.

---

## Project structure

```
mini_agent/
├── agent.py              # CLI entry point — argument parsing, REPL
├── core.py               # The agent loop: run_agent()
├── reliability.py        # Reliability layer: retry, circuit breaker, validation, tracing, provider fallback
├── providers.py          # hailo-ollama provider (stdlib HTTP, native or prompt tool calling)
├── tools.py              # Tool implementations and schemas
├── ui.py                 # Spinner — thread-safe braille activity indicator
├── mcp_server.py         # Demo MCP server (to_uppercase, count_words)
├── mcp_client.py         # MCP client helper — spawns the server, calls tools via JSON-RPC
├── requirements.txt
├── .env.example
├── deployment-pipeline/          # see "Deployment Pipeline" section below
│   ├── release.py                # iOS + Android release automation (build → sign → upload)
│   ├── infra_tools.py            # agent-callable wrappers: deploy, test, commit, trigger
│   ├── ExportOptions.plist       # iOS export options template (edit teamID)
│   └── scripts/
│       ├── deploy.sh             # one-command web deploy (Firebase Hosting)
│       ├── setup.sh              # one-time setup check
│       ├── test.sh               # run pytest
│       ├── versions.cfg          # build number tracking (auto-bumped by release.py)
│       └── cd_secrets.env.example  # secrets template — copy to cd_secrets.env
└── visual-testing/               # see visual-testing/README.md for full setup & usage
    ├── ui_agent.py           # Visual flow testing — record & check modes (Anthropic Claude or OpenAI, picked by --model)
    ├── ui_agent_local.py     # Same tool, local Ollama backend (zero cloud cost)
    ├── mobile_tools.py       # iOS Simulator + Android screenshot / tap / swipe helpers
    ├── requirements-ui.txt       # anthropic, openai, pillow
    ├── requirements-ui-local.txt # openai (Ollama client), pillow
    ├── baselines-ios/            # example recorded flow — one per platform, never shared
    │   ├── index.json            #   navigation map: label, advance gesture, assertions per step
    │   └── myapp_onboarding_step_*.png
    └── baselines-android/        # same flow name, recorded independently on Android
        ├── index.json
        └── myapp_onboarding_step_*.png
```

### Adding your own tools

1. Write a Python function in `tools.py`
2. Add it to `TOOL_FUNCTIONS`
3. Add its JSON schema to `TOOL_SCHEMAS`

The `web_search` and `get_weather` tools are stubs. Replace them with real API calls (Brave Search, Tavily, OpenWeatherMap, etc.) to make the agent genuinely useful.

---

## How the loop works

```
┌─────────────────────────────────────────┐
│  messages = [system, user_task]         │
│                                         │
│  while True:                            │
│    response = llm.complete(messages)    │
│                                         │
│    if no tool_calls:                    │
│      return response.content  ← done   │
│                                         │
│    for each tool_call:                  │
│      result = call_tool(name, args)     │
│      messages.append(tool_result)       │
└─────────────────────────────────────────┘
```

---

## MCP tools

`mcp_server.py` is a standalone [Model Context Protocol](https://modelcontextprotocol.io) server that exposes two tools — `to_uppercase` and `count_words`. The agent calls them transparently via `mcp_client.py`; from the agent's perspective they are no different from any other tool.

**Verify the server starts:**

```bash
python mcp_server.py
```

It will block waiting for JSON-RPC messages on stdin — that's expected. Press `Ctrl+C` to exit. Normally the agent spawns it automatically as a subprocess.

**How the communication works:**

```
agent (tools.py)
  └── mcp_client.py          # asyncio JSON-RPC client
        └── subprocess: mcp_server.py   # FastMCP server on stdio
```

Each tool call spawns a fresh subprocess, performs the `initialize` → `call_tool` handshake, and exits. To add your own MCP tools, define them in `mcp_server.py` with `@mcp.tool()` and register wrapper functions in `tools.py` following the same pattern as `mcp_to_uppercase` and `mcp_count_words`.

---

## Notes

- `calculate()` uses Python's `eval()` with empty builtins — safe enough for a demo, not for production. Replace with a proper math library (`sympy`, `asteval`) for real use.
- `visual-testing/` and `deployment-pipeline/` are unchanged from upstream and still use their own cloud / Ollama clients (see their READMEs).

---

## Changelog

### hailo fork — hailo-ollama only

- `providers.py` is a single `HailoOllamaProvider` on `/api/chat` (stdlib `urllib`, no SDK).
- Native tool calling (HailoRT 5.5.0+), prompt-based fallback, `<think>` stripping.
- Default model `qwen3:1.7b`; `qwen3:4b` for 8 GB devices.
- Removed OpenAI, Anthropic, Gemini, plain Ollama, `--mode remote/mixed`, and `ask_remote()`.

### v0.5 — Deployment pipeline

Added `deployment-pipeline/` — release automation and agent-callable CI/CD tools.

- `release.py` — one command to build, sign, and upload iOS (TestFlight) or Android (Play Store open testing track)
- `infra_tools.py` — `run_tests`, `git_status`, `git_commit_push`, `deploy_web`, `deploy_mobile`, `remote_trigger` as agent-callable tools following the same TOOL_FUNCTIONS / TOOL_SCHEMAS pattern as `tools.py`
- `scripts/deploy.sh`, `test.sh`, `setup.sh` — shell scripts the Python wrappers delegate to
- Build numbers tracked automatically in `scripts/versions.cfg`; credentials isolated in `scripts/cd_secrets.env` (gitignored)

### v0.4 — OpenAI backend for ui_agent.py

`ui_agent.py` now supports OpenAI as a second cloud backend alongside Anthropic Claude. `_vision_call()` dispatches on the `--model` value: a name starting with `gpt-` (e.g. `gpt-4o`) routes to OpenAI's chat completions API with `image_url` content blocks; anything else routes to Anthropic, as before. `requirements-ui.txt` now installs `openai` alongside `anthropic` and `pillow`.

### v0.3 — Visual testing agent

Added `visual-testing/` — a standalone LLM-driven visual flow testing tool for iOS Simulator and Android. No pixel diff, no XPath selectors — the LLM judges screens by visual intent.

**Core tool (`ui_agent.py`)**
- `record` mode: human navigates step by step, LLM labels each screen and stores the advancement gesture (tap coordinates or swipe direction) per step in `baselines-<platform>/index.json` — iOS and Android baselines are kept in separate directories, so the same flow name can be recorded on both without collision
- `check` mode: LLM drives the device autonomously — on MATCH applies the stored gesture (zero extra LLM calls); on MISMATCH falls back to dynamic gesture inference, retries up to `--max-retries`
- `check-all` mode: runs every recorded flow, prints a combined pass/fail summary, exits `1` on any failure — CI-ready
- `--describe` flag: plain-English flow spec threaded into all three LLM prompts (capture, match, tap) so the LLM knows navigation intent and per-screen assertions

**Local Ollama backend (`ui_agent_local.py`)**
- Zero cloud cost for check runs — routes all vision calls to a local Ollama model (`llama3.2-vision:11b` default)
- Hard tap Y clamp at 85% (`_MAX_TAP_Y`) to avoid the iOS home indicator gesture area
- Markdown bold stripping (`**LABEL:**`) for local models that add formatting

**iOS Simulator gestures (`mobile_tools.py`)**
- `tap_simulator_osx`: CoreGraphics `CGEventPost` via JXA — activate + tap in one atomic script to prevent Terminal focus stealing
- `swipe_simulator_osx`: 10 drag steps so iOS registers the motion as a real swipe gesture

**Android improvements**
- `_get_android_screen_size()`: reads live resolution from `adb shell wm size` — no hardcoded dimensions
- `tap_android()`: extracted to `mobile_tools.py`, uses dynamic resolution
- `swipe_android()`: fixed to use dynamic resolution

### v0.2 — Reliability layer

Added `reliability.py` on top of the unchanged core loop. Every item is independently useful; none require changes to `core.py`.

**LLM-level resilience**
- `_RetryingProvider` — wraps any provider's `complete()` with exponential backoff + jitter (up to 5 attempts). The agent loop never sees a transient network failure.

**Tool-level resilience**
- `with_retry()` — retries a single tool call on exception (configurable attempts and base delay)
- `CircuitBreaker` — stops calling a broken tool after N consecutive failures; auto-resets after a timeout
- `validated_call()` — validates tool arguments against the JSON schema with pydantic before execution; returns an error string the model can self-correct on
- `traced_call()` — emits a structured log line (tool name, args, result, duration) for every call regardless of outcome
- `run_agent_reliable()` — `run_agent()` with all four tool layers stacked; drop-in replacement

**Provider fallback**
- `run_agent_with_fallback()` — tries a list of providers in order, falling back on exception *or* silent failure (model outputs tool invocations as plain text instead of structured `tool_calls`)
- `_looks_like_failed_tool_call()` — heuristic that turns the silent failure into a detectable error
- `--fallback MODEL [MODEL …]` CLI flag — e.g. `--fallback llama3.1 mistral-nemo`

**agent.py** — `run_agent_reliable()` is now the default for `local` and `remote` modes.

### v0.1 — Initial release

Core agent loop: `agent.py`, `core.py`, `providers.py`, `tools.py`, `ui.py`, MCP client/server.

---

## Deployment Pipeline

`deployment-pipeline/` contains the tools to close the loop so the agent can deploy without you at the keyboard.

**Quick start:**
```bash
cd deployment-pipeline

# 1. Check dependencies
bash scripts/setup.sh

# 2. Fill in credentials
cp scripts/cd_secrets.env.example scripts/cd_secrets.env
# edit cd_secrets.env

# 3. Edit project paths at the top of release.py

# 4. Validate credentials without building
python3 release.py --ios --dry-run
python3 release.py --android --dry-run

# 5. Ship
python3 release.py --ios
python3 release.py --android
bash scripts/deploy.sh          # web
```

**iOS tip:** Add this key to your app's `Info.plist` so TestFlight skips the encryption compliance question — builds go live immediately after upload:
```xml
<key>ITSAppUsesNonExemptEncryption</key>
<false/>
```

**Wiring infra tools into the agent:**
```python
from deployment_pipeline.infra_tools import TOOL_FUNCTIONS as INFRA_FN
from deployment_pipeline.infra_tools import TOOL_SCHEMAS   as INFRA_SCHEMAS

# merge with your existing tools
all_tools     = TOOL_SCHEMAS + INFRA_SCHEMAS
all_functions = {**TOOL_FUNCTIONS, **INFRA_FN}
```

Available agent tools: `run_tests`, `git_status`, `git_commit_push`, `deploy_web`, `deploy_mobile`, `remote_trigger`.

Full setup guide, iOS and Android one-time setup, command reference, and gotchas: **[deployment-pipeline/README.md](deployment-pipeline/README.md)**

---

## Visual Testing

`visual-testing/` is a standalone LLM-driven visual flow testing agent for iOS Simulator and Android. **No pixel comparison, no coordinate hunting** — the LLM looks at screenshots and judges by visual intent.

**Quick start:**
```bash
cd visual-testing
pip install -r requirements-ui.txt   # cloud (Anthropic Claude or OpenAI)
# or:
pip install -r requirements-ui-local.txt  # local (Ollama, zero cost)

python ui_agent.py record --ios "myapp-onboarding"
python ui_agent.py check  --ios "myapp-onboarding"
python ui_agent.py check  --ios --model gpt-4o "myapp-onboarding"  # or use OpenAI
python ui_agent.py check-all --ios && ./deploy.sh
```

Full setup guide, iOS Simulator configuration, Android setup, local model comparison, and command reference: **[visual-testing/README.md](visual-testing/README.md)**

> **Disclaimer:** this is a working proof-of-concept, not a production tool. It's here to show the idea is real and — hopefully — to inspire someone to take it further. If you have the desire or the funding to build something serious on top of it, I'd love to be involved. Reach me on [LinkedIn](https://www.linkedin.com/in/sergey-neskoromny/).

---

## License

MIT
