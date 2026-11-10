# Mini Agent

Companion repo for the series Software Engineering in the Agentic AI Era. No agent framework: Python, the standard library, and a `while` loop. This fork runs every LLM call on a Hailo-10H through hailo-ollama (`qwen3:1.7b` default).

## Always

- Do not commit `.env` or credentials.
- Add tools only through `tools.py` (`TOOL_FUNCTIONS` + `TOOL_SCHEMAS`).
- If the task matches a skill, load the skill before acting.

## Skills

- `running-the-agent`: venv, `.env`, hailo-ollama, tool modes, MCP, fallback
- `visual-check`: `visual-testing/ui_agent.py` record and check
- `deploy`: `deployment-pipeline/release.py` and `infra_tools.py`

## Tools

Claude Code reads `CLAUDE.md` (same text as this file). Gemini CLI reads `GEMINI.md`, which points here. Keep `CLAUDE.md` and `AGENTS.md` in sync by hand until you make one a generated copy of the other.

<!-- mirror of CLAUDE.md. Keep them in sync. -->

