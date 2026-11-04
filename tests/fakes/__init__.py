"""Fake harnesses for the conformance suite (WP6).

`FakeWorld` stands in for every external thing an adapter touches:
`shutil.which`, `subprocess.run`, `urllib.request.urlopen`. It answers
like each real harness does on the happy path and can be switched into
`timeout` or `unavailable` mode. No real binary or socket is ever used.
"""

from __future__ import annotations

import io
import json
import subprocess
import urllib.error
from pathlib import Path

# The canonical schema every conformance call uses. `note` is optional so
# strict-mode adapters must emit it as null and the suite can check the
# null is stripped again.
CONFORMANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "note": {"type": "string"},
    },
    "required": ["findings", "confidence"],
}
GOOD_PAYLOAD = {"findings": "ok", "confidence": 0.9}
GOOD_PAYLOAD_WITH_NULL = {"findings": "ok", "confidence": 0.9, "note": None}
PLAIN_TEXT = "The answer is forty-two."


class _Resp:
    def __init__(self, body: dict, status: int = 200):
        self._body = json.dumps(body).encode("utf-8")
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeWorld:
    """mode: "ok" | "timeout" | "unavailable"."""

    def __init__(self, mode: str = "ok"):
        self.mode = mode
        self.spawned: list[list[str]] = []
        self.requests: list[str] = []

    # -- shutil.which -----------------------------------------------------
    def which(self, name: str):
        if self.mode == "unavailable":
            return None
        return f"/fake/bin/{name}"

    # -- subprocess.run ---------------------------------------------------
    def run(self, cmd, **kw):
        if cmd[:2] == ["claude", "--help"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="--tools --max-turns --effort --json-schema", stderr="")
        self.spawned.append(list(cmd))
        if self.mode == "timeout":
            raise subprocess.TimeoutExpired(cmd, kw.get("timeout", 1))
        tool = Path(cmd[0]).name
        prompt = kw.get("input") or ""
        if tool == "codex":
            return self._codex(cmd)
        if tool == "claude":
            return self._claude(cmd, prompt)
        if tool == "gemini":
            return self._gemini(cmd)
        return self._generic(cmd, prompt)

    def _wants_schema_in_prompt(self, cmd, prompt: str) -> bool:
        return "Output contract" in prompt or any("Output contract" in c for c in cmd)

    def _codex(self, cmd):
        out = Path(cmd[cmd.index("--output-last-message") + 1])
        if "--output-schema" in cmd:
            out.write_text(json.dumps(GOOD_PAYLOAD_WITH_NULL), encoding="utf-8")
        else:
            out.write_text(PLAIN_TEXT, encoding="utf-8")
        events = json.dumps({"type": "turn.completed", "usage": {"input_tokens": 100, "cached_input_tokens": 20, "output_tokens": 10}})
        return subprocess.CompletedProcess(cmd, 0, stdout=events + "\n", stderr="")

    def _claude(self, cmd, prompt):
        env = {"type": "result", "is_error": False, "total_cost_usd": 0.002,
               "usage": {"input_tokens": 50, "cache_read_input_tokens": 500, "output_tokens": 20}}
        if "--json-schema" in cmd:
            env["structured_output"] = dict(GOOD_PAYLOAD)
            env["result"] = json.dumps(GOOD_PAYLOAD)
        else:
            env["result"] = PLAIN_TEXT
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(env), stderr="")

    def _gemini(self, cmd):
        prompt = cmd[cmd.index("-p") + 1]
        text = json.dumps(GOOD_PAYLOAD) if "Output contract" in prompt else PLAIN_TEXT
        env = {"response": text, "stats": {"input_tokens": 30, "output_tokens": 8}}
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(env), stderr="")

    def _generic(self, cmd, prompt):
        # the template may hand the prompt over as a file instead of stdin
        for c in cmd:
            if c.endswith("prompt.md") and Path(c).exists():
                prompt = Path(c).read_text(encoding="utf-8")
        schema_native = any(c.endswith("schema.json") for c in cmd)
        text = json.dumps(GOOD_PAYLOAD) if (schema_native or "Output contract" in prompt) else PLAIN_TEXT
        return subprocess.CompletedProcess(cmd, 0, stdout=text, stderr="")

    # -- urllib.request.urlopen ------------------------------------------
    def urlopen(self, req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        self.requests.append(url)
        if self.mode == "unavailable":
            raise urllib.error.URLError("connection refused")
        if url.endswith("/models") or url.endswith("/api/tags"):
            return _Resp({"data": []})
        if self.mode == "timeout":
            raise TimeoutError()
        body = json.loads(req.data.decode("utf-8")) if getattr(req, "data", None) else {}
        if url.endswith("/chat/completions"):
            content = json.dumps(GOOD_PAYLOAD_WITH_NULL) if "response_format" in body else PLAIN_TEXT
            return _Resp({"choices": [{"message": {"content": content}}],
                          "usage": {"prompt_tokens": 40, "completion_tokens": 9}})
        if url.endswith("/api/generate"):
            content = json.dumps(GOOD_PAYLOAD) if "format" in body else PLAIN_TEXT
            return _Resp({"response": content, "done": True, "prompt_eval_count": 33, "eval_count": 7})
        return _Resp({}, 404)


def install(monkeypatch, world: FakeWorld) -> None:
    """Point every adapter module at `world`."""
    import shutil
    import urllib.request
    from backends.adapters import cli, generic_cli, ollama, openai_http

    for mod in (cli, generic_cli):
        monkeypatch.setattr(mod.shutil, "which", world.which)
        monkeypatch.setattr(mod.subprocess, "run", world.run)
    for mod in (ollama, openai_http):
        monkeypatch.setattr(mod.urllib.request, "urlopen", world.urlopen)
    monkeypatch.setattr(shutil, "which", world.which)
    monkeypatch.setattr(urllib.request, "urlopen", world.urlopen)
    # the ACP adapter spawns tests/fakes/acp_agent.py for real; it reads the mode from env
    from backends.adapters import acp as acp_mod
    monkeypatch.setattr(acp_mod.shutil, "which", world.which)
    monkeypatch.setenv("ACP_FAKE_MODE", world.mode)
