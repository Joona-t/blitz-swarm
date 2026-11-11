"""Ollama adapter: `think` and `max_parallel` provider options (2026-11-11,
found setting up a local Qwen run: a thinking model counts its reasoning as
output tokens, and a one-slot server makes a swarm's parallel calls wait
inside the request timeout)."""

from __future__ import annotations

import json
import threading
import time

from backends import AgentCall
from backends.adapters import ollama as O
from backends.registry import backend_from_provider
from config import load_config


class _Resp:
    status = 200

    def __init__(self, body):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self._body).encode()


def _call(**kw):
    return AgentCall(role="researcher", prompt="hi", timeout_s=5, **kw)


def test_think_is_sent_only_when_configured(monkeypatch):
    bodies = []

    def urlopen(req, timeout):
        bodies.append(json.loads(req.data.decode()))
        return _Resp({"response": "answer", "thinking": "hmm" * 10, "eval_count": 9, "done": True, "done_reason": "stop"})

    monkeypatch.setattr(O.urllib.request, "urlopen", urlopen)
    r = O.OllamaHTTPBackend(model="m").call(_call())
    assert "think" not in bodies[0], "None leaves the server default alone"
    assert r.text == "answer" and r.metadata["thinking_chars"] == 30 and r.metadata["done_reason"] == "stop"
    O.OllamaHTTPBackend(model="m", think=False).call(_call())
    O.OllamaHTTPBackend(model="m", think=True).call(_call())
    assert bodies[1]["think"] is False and bodies[2]["think"] is True


def test_max_parallel_serialises_calls_and_waiting_is_not_call_time(monkeypatch):
    state = {"in": 0, "peak": 0}
    lock = threading.Lock()
    barrier = {"b": None}

    def urlopen(req, timeout):
        with lock:
            state["in"] += 1
            state["peak"] = max(state["peak"], state["in"])
        if barrier["b"] is not None:
            barrier["b"].wait(timeout=10)      # returns only when all four are inside together
        else:
            time.sleep(0.05)
        with lock:
            state["in"] -= 1
        return _Resp({"response": "ok", "eval_count": 1, "done": True})

    monkeypatch.setattr(O.urllib.request, "urlopen", urlopen)

    def run(backend_factory):
        state["peak"] = 0
        results = []
        ts = [threading.Thread(target=lambda: results.append(backend_factory().call(_call()))) for _ in range(4)]
        t0 = time.monotonic()
        [t.start() for t in ts]
        [t.join() for t in ts]
        return results, time.monotonic() - t0

    # Known-bad: without the gate all four calls are in the server at once.
    barrier["b"] = threading.Barrier(4)
    results, _ = run(lambda: O.OllamaHTTPBackend(model="m", base_url="http://gate-test-a"))
    assert state["peak"] == 4 and not any(r.errored for r in results)
    # Separate backend objects for the same server share one gate.
    barrier["b"] = None
    results, wall = run(lambda: O.OllamaHTTPBackend(model="m", base_url="http://gate-test-b", max_parallel=1))
    assert state["peak"] == 1
    assert max(r.metadata["queued_s"] for r in results) >= 0.15, "the last caller waited for three 0.05 s calls"
    # Waiting and calling are disjoint stretches of the run. If queue time were
    # billed to the call (and so to its timeout) the last caller would break this.
    assert all(r.elapsed_s + r.metadata["queued_s"] <= wall + 0.01 for r in results)


def test_gate_is_released_when_the_request_fails(monkeypatch):
    def boom(req, timeout):
        raise TimeoutError()

    monkeypatch.setattr(O.urllib.request, "urlopen", boom)
    b = O.OllamaHTTPBackend(model="m", base_url="http://gate-test-c", max_parallel=1)
    assert b.call(_call()).errored and b.call(_call()).errored, "a second call must not hang on a leaked slot"


def test_options_come_from_the_provider_table(tmp_path):
    p = tmp_path / "blitz.toml"
    p.write_text('''
[models]
default = "standard"
[backend]
default = "ollama"
[backend.providers.ollama]
adapter = "ollama_http"
base_url = "http://127.0.0.1:11434"
think = false
max_parallel = 1
[backend.providers.ollama.tiers]
standard = "some-model"
''', encoding="utf-8")
    cfg = load_config(p)
    inner = backend_from_provider("ollama", cfg.backend.get_provider("ollama")).inner
    assert inner.think is False and inner.max_parallel == 1
