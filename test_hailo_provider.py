"""Offline tests for HailoOllamaProvider against a fake hailo-ollama /api/chat server."""

import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from core import run_agent
from providers import HailoOllamaProvider, _normalize_host, _parse_text_tool_call, create_provider
from tools import TOOL_SCHEMAS

_CALC_ARGS = {"expression": "847 * 0.15"}
_FINAL = "15% of 847 is 127.05."


class _FakeHailo(BaseHTTPRequestHandler):
    """Behaves like hailo-ollama 5.5.0. Knobs set per test on the class."""

    requests: list[dict] = []
    reject_tools = False        # answer 400 "does not support tools" when `tools` is sent
    string_args = False         # return tool arguments as a JSON string
    think = False               # prefix replies with a <think> block

    def log_message(self, *args):
        pass

    def _reply(self, status: int, obj) -> None:
        out = (json.dumps(obj) if not isinstance(obj, str) else obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_POST(self):
        cls = type(self)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cls.requests.append(body)
        if "tools" in body and cls.reject_tools:
            return self._reply(400, f"{body['model']} does not support tools")

        last = body["messages"][-1]
        got_result = last["role"] == "tool" or last["content"].startswith("TOOL RESULT")
        prefix = "<think>let me compute</think>\n" if cls.think else ""
        if got_result:
            msg = {"role": "assistant", "content": prefix + _FINAL}
        elif "tools" in body:
            args = json.dumps(_CALC_ARGS) if cls.string_args else _CALC_ARGS
            msg = {"role": "assistant", "content": prefix,
                   "tool_calls": [{"function": {"name": "calculate", "arguments": args}}]}
        else:
            msg = {"role": "assistant",
                   "content": prefix + '```json\n{"tool": "calculate", "arguments": {"expression": "847 * 0.15"}}\n```'}
        self._reply(200, {"model": body["model"], "message": msg, "done": True})


class HailoProviderTest(unittest.TestCase):
    def setUp(self):
        _FakeHailo.requests = []
        _FakeHailo.reject_tools = _FakeHailo.string_args = _FakeHailo.think = False
        self.server = HTTPServer(("127.0.0.1", 0), _FakeHailo)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.host = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def _run(self, **kw) -> str:
        p = HailoOllamaProvider(host=self.host, **kw)
        self.provider = p
        return run_agent("What is 15% of 847?", p, verbose=False)

    def test_native_mode_is_default(self):
        self.assertEqual(self._run(), _FINAL)
        first, second = _FakeHailo.requests
        self.assertEqual(first["model"], "qwen3:1.7b")
        self.assertFalse(first["stream"])
        self.assertIn("tools", first)
        self.assertEqual(second["messages"][-1],
                         {"role": "tool", "tool_name": "calculate", "content": "127.05"})
        self.assertEqual(second["messages"][-2]["tool_calls"][0]["function"]["arguments"], _CALC_ARGS)

    def test_prompt_mode(self):
        self.assertEqual(self._run(tool_mode="prompt"), _FINAL)
        first, second = _FakeHailo.requests
        self.assertNotIn("tools", first)
        self.assertIn("calculate", first["messages"][0]["content"])
        self.assertTrue(second["messages"][-1]["content"].startswith("TOOL RESULT (calculate): 127.05"))

    def test_switches_to_prompt_when_model_rejects_tools(self):
        _FakeHailo.reject_tools = True
        self.assertEqual(self._run(model="gemma4:e2b"), _FINAL)
        self.assertEqual(self.provider.tool_mode, "prompt")
        self.assertIn("tools", _FakeHailo.requests[0])
        self.assertNotIn("tools", _FakeHailo.requests[1])

    def test_string_arguments_and_think_blocks(self):
        _FakeHailo.string_args = _FakeHailo.think = True
        self.assertEqual(self._run(), _FINAL)

    def test_parse_ignores_unknown_and_plain_text(self):
        known = {t["function"]["name"] for t in TOOL_SCHEMAS}
        self.assertIsNone(_parse_text_tool_call("The answer is 4.", known))
        self.assertIsNone(_parse_text_tool_call('{"tool": "rm_rf", "arguments": {}}', known))
        self.assertEqual(
            _parse_text_tool_call('Sure: {"name": "get_current_date", "arguments": {}}', known),
            ("get_current_date", {}),
        )

    def test_host_env_and_normalization(self):
        self.assertEqual(_normalize_host("0.0.0.0:8000"), "http://localhost:8000")
        self.assertEqual(_normalize_host(":9000"), "http://localhost:9000")
        with mock.patch.dict(os.environ, {"OLLAMA_HOST": "pi.local:8000"}, clear=True):
            self.assertEqual(create_provider().host, "http://pi.local:8000")
        with mock.patch.dict(os.environ, {"HAILO_OLLAMA_HOST": "http://a:1", "OLLAMA_HOST": "b:2"}, clear=True):
            self.assertEqual(create_provider("qwen3:4b").host, "http://a:1")

    def test_unreachable_server(self):
        p = HailoOllamaProvider(host="http://127.0.0.1:1", timeout=2)
        with self.assertRaises(ConnectionError):
            p.complete([{"role": "user", "content": "hi"}], [])


if __name__ == "__main__":
    unittest.main()
