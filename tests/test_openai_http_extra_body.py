"""openai_http: `extra_body` provider option and finish/thinking metadata
(2026-11-12, for a vLLM server whose model thinks by default)."""

from __future__ import annotations

import json

from backends import AgentCall
from backends.adapters import openai_http as H
from backends.registry import backend_from_provider
from bench.runner import _provider_options
from config import load_config

TOML = '''
[models]
default = "standard"
[backend]
default = "local"
[backend.providers.local]
adapter = "openai_http"
base_url = "http://127.0.0.1:9/v1"
[backend.providers.local.extra_body]
model = "must-not-win"
[backend.providers.local.extra_body.chat_template_kwargs]
enable_thinking = false
[backend.providers.local.tiers]
standard = "served-name"
'''


class _Resp:
    status = 200

    def __init__(self, body):
        self._b = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._b


def test_extra_body_reaches_the_request_and_cannot_replace_owned_fields(monkeypatch, tmp_path):
    p = tmp_path / "blitz.toml"
    p.write_text(TOML, encoding="utf-8")
    cfg = load_config(p)
    sent = []

    def urlopen(req, timeout):
        sent.append(json.loads(req.data.decode()))
        return _Resp({"choices": [{"finish_reason": "length",
                                   "message": {"content": "answer", "reasoning_content": "x" * 40}}],
                      "usage": {"prompt_tokens": 3, "completion_tokens": 9}})

    monkeypatch.setattr(H.urllib.request, "urlopen", urlopen)
    backend = backend_from_provider("local", cfg.backend.get_provider("local"))
    r = backend.call(AgentCall(role="researcher", prompt="hi", model="served-name", timeout_s=5))
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent[0]["model"] == "served-name", "extra_body must not override the resolved model"
    assert r.text == "answer" and r.output_tokens == 9
    assert r.metadata["finish_reason"] == "length" and r.metadata["thinking_chars"] == 40
    # Known-bad: without the option the field is absent and the server default (thinking on) applies.
    H.OpenAIHTTPBackend(base_url="http://127.0.0.1:9/v1", model="m").call(AgentCall(role="r", prompt="hi", timeout_s=5))
    assert "chat_template_kwargs" not in sent[1]


def test_run_record_keeps_nested_provider_options(tmp_path):
    p = tmp_path / "blitz.toml"
    p.write_text(TOML, encoding="utf-8")
    opts = _provider_options(load_config(p), "local")
    assert opts["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
