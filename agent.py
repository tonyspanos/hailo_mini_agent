#!/usr/bin/env python3
"""
Mini Agent — a minimal AI agent with no framework dependencies.

Every LLM call goes to hailo-ollama on a Hailo-10H (Raspberry Pi AI HAT+ 2),
default http://localhost:8000. Tools run locally.

Usage examples:
  python agent.py "What is 15% of 847?"
  python agent.py --model qwen3:4b "What's the weather in Tokyo and today's date?"
  python agent.py --fallback qwen3:4b qwen2.5:1.5b "What is 144 * 37?"
  python agent.py --interactive
"""

from __future__ import annotations

import argparse
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; export env vars manually if not installed

from providers import DEFAULT_MODEL, create_provider, LLMProvider
from reliability import run_agent_reliable, run_agent_with_fallback


_BANNER = """\
╔══════════════════════════════════════════╗
║           Mini Agent  v0.1               ║
║        hailo-ollama  ·  Hailo-10H        ║
╚══════════════════════════════════════════╝"""


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent",
        description="A minimal AI agent on a Hailo-10H. No frameworks, just the loop.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        metavar="MODEL",
        help=f"hailo-ollama model (default: {DEFAULT_MODEL}). "
             "qwen3:4b needs the 8 GB Hailo-10H. List models with GET /hailo/v1/list.",
    )
    p.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start an interactive REPL session instead of running a single task",
    )
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress tool call trace output",
    )
    p.add_argument(
        "--fallback",
        nargs="+",
        default=[],
        metavar="MODEL",
        help="Fallback hailo-ollama model(s) tried in order if the primary fails "
             "(e.g. --fallback qwen3:4b qwen2.5:1.5b).",
    )
    p.add_argument(
        "task",
        nargs="*",
        help="Task to run (omit when using --interactive)",
    )
    return p


def run_task(task: str, args, provider: LLMProvider) -> str:
    verbose = not args.quiet
    if args.fallback:
        providers = [provider] + [create_provider(m) for m in args.fallback]
        return run_agent_with_fallback(task, providers, verbose=verbose)
    return run_agent_reliable(task, provider, verbose=verbose)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    print(_BANNER)

    provider = create_provider(args.model)
    print(f"\nModel: {provider}  @  {provider.host}  (tools: {provider.tool_mode})\n")

    if args.interactive:
        _repl(args, provider)
        return

    if not args.task:
        parser.print_help()
        sys.exit(0)
    task = " ".join(args.task)
    print(f"Task: {task}\n")
    try:
        answer = run_task(task, args, provider)
        print(f"\nAnswer: {answer}\n")
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}\n")
        sys.exit(1)


def _repl(args, provider: LLMProvider) -> None:
    print("Interactive mode. Type your task and press Enter.")
    print("Commands: 'exit' or Ctrl+C to quit.\n")

    while True:
        try:
            task = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if task.lower() in ("exit", "quit", "q", ":q"):
            print("Goodbye.")
            break
        if not task:
            continue

        print()
        try:
            answer = run_task(task, args, provider)
            print(f"\nAgent: {answer}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
