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


def test_empty_content_is_an_error_not_the_raw_envelope(monkeypatch):
    """2026-11-12: a reply with choices but no content (all tokens spent on
    reasoning, cut at the limit) came back with text = the raw JSON body, so a
    bench arm saved the envelope as the report and called it a success."""
    body = {"choices": [{"finish_reason": "length",
                         "message": {"content": None, "reasoning_content": "thinking " * 50}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 900}}
    monkeypatch.setattr(H.urllib.request, "urlopen", lambda req, timeout: _Resp(body))
    r = H.OpenAIHTTPBackend(base_url="http://127.0.0.1:9/v1", model="m").call(AgentCall(role="r", prompt="hi", timeout_s=5))
    assert r.errored and "empty content" in r.error and "length" in r.error
    assert r.text == "" and "choices" not in r.text
    assert r.output_tokens == 900 and r.metadata["thinking_chars"] == len("thinking " * 50)
    assert r.metadata["reasoning_tokens"] is None          # not reported by this body
    # A body that is not the chat shape still surfaces as text (unchanged).
    monkeypatch.setattr(H.urllib.request, "urlopen", lambda req, timeout: _Resp({"weird": 1}))
    r2 = H.OpenAIHTTPBackend(base_url="http://127.0.0.1:9/v1", model="m").call(AgentCall(role="r", prompt="hi", timeout_s=5))
    assert '"weird"' in r2.text


def test_reasoning_tokens_from_vllm_usage_details(monkeypatch):
    """vLLM counts thinking inside completion_tokens and reports the split."""
    body = {"choices": [{"finish_reason": "stop", "message": {"content": "ok", "reasoning": "r" * 426}}],
            "usage": {"prompt_tokens": 62, "completion_tokens": 133,
                      "completion_tokens_details": {"reasoning_tokens": 91}}}
    monkeypatch.setattr(H.urllib.request, "urlopen", lambda req, timeout: _Resp(body))
    r = H.OpenAIHTTPBackend(base_url="http://127.0.0.1:9/v1", model="m").call(AgentCall(role="r", prompt="hi", timeout_s=5))
    assert r.output_tokens == 133 and r.metadata["reasoning_tokens"] == 91 and r.metadata["thinking_chars"] == 426


def test_max_parallel_caps_requests_in_flight(monkeypatch, tmp_path):
    """2026-11-12 (vLLM live run): the swarm's selector fanned out 32 requests at
    once against a server shared with another client that owns half its slots.
    `max_parallel` in the provider table caps this process's requests in flight,
    shared across backend objects for the same server (mirrors ollama_http)."""
    import threading
    import time as _t
    p = tmp_path / "blitz.toml"
    p.write_text(TOML.replace('base_url = "http://127.0.0.1:9/v1"',
                              'base_url = "http://127.0.0.1:9/v1"\nmax_parallel = 3'), encoding="utf-8")
    cfg = load_config(p)
    lock = threading.Lock()
    state = {"now": 0, "peak": 0}

    def urlopen(req, timeout):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        _t.sleep(0.05)
        with lock:
            state["now"] -= 1
        return _Resp({"choices": [{"finish_reason": "stop", "message": {"content": "a"}}], "usage": {}})

    monkeypatch.setattr(H.urllib.request, "urlopen", urlopen)
    # Two backend objects for the same server share one gate.
    b1 = backend_from_provider("local", cfg.backend.get_provider("local"))
    b2 = backend_from_provider("local", cfg.backend.get_provider("local"))
    assert b1.inner.max_parallel == 3
    results = []
    threads = [threading.Thread(target=lambda b=b: results.append(
        b.call(AgentCall(role="r", prompt="hi", model="served-name", timeout_s=5)))) for b in [b1, b2] * 6]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(results) == 12 and not any(r.errored for r in results)
    assert state["peak"] == 3, state
    assert any(r.metadata["queued_s"] > 0 for r in results)
    assert "max_parallel" not in b1.inner.metadata
    # Known-bad: without the option all 12 go at once.
    state["peak"] = 0
    b0 = H.OpenAIHTTPBackend(base_url="http://127.0.0.1:9/v1", model="m")
    threads = [threading.Thread(target=lambda: b0.call(AgentCall(role="r", prompt="hi", timeout_s=5))) for _ in range(12)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert state["peak"] > 3
