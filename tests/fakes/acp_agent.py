"""A fake ACP agent for tests: JSON-RPC 2.0 over stdio, protocol v1.

Behaviour is picked by ACP_FAKE_MODE:
  ok          answer end_turn; JSON payload if the prompt carries an
              "Output contract", else prose
  permission  ask session/request_permission for a tool first; print
              "TOOL RAN" if allowed, "DENIED" otherwise
  refusal     stopReason=refusal
  hang        never answer the prompt (timeout path)
  crash       exit 3 after initialize
  timeout     alias of hang (conformance FakeWorld mode name)
  unavailable alias of ok (conformance handles availability via which())
"""

from __future__ import annotations

import json
import os
import sys
import time

MODE = os.environ.get("ACP_FAKE_MODE", "ok")
if MODE in ("timeout",):
    MODE = "hang"
if MODE in ("unavailable",):
    MODE = "ok"

GOOD = {"findings": "ok", "confidence": 0.9}
PLAIN = "The answer is forty-two."


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def notify(session_id, update):
    send({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": session_id, "update": update}})


def read():
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    return json.loads(line)


def main():
    pending_perm = None
    while True:
        msg = read()
        method = msg.get("method")
        rid = msg.get("id")
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": rid, "result": {
                "protocolVersion": 1,
                "agentCapabilities": {"loadSession": False},
                "authMethods": [],
            }})
            if MODE == "crash":
                sys.stderr.write("fake agent: crashing on purpose\n")
                sys.stderr.flush()
                sys.exit(3)
        elif method == "session/new":
            send({"jsonrpc": "2.0", "id": rid, "result": {"sessionId": "sess-1"}})
        elif method == "session/prompt":
            sid = msg["params"]["sessionId"]
            prompt = "".join(b.get("text", "") for b in msg["params"]["prompt"])
            if MODE == "hang":
                time.sleep(60)
                continue
            if MODE == "refusal":
                send({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "refusal"}})
                continue
            if MODE == "permission":
                pending_perm = (rid, sid, prompt)
                send({"jsonrpc": "2.0", "id": 900, "method": "session/request_permission", "params": {
                    "sessionId": sid,
                    "toolCall": {"toolCallId": "tc1", "title": "Run shell: ls", "kind": "execute", "status": "pending"},
                    "options": [
                        {"optionId": "a1", "name": "Allow once", "kind": "allow_once"},
                        {"optionId": "r1", "name": "Reject once", "kind": "reject_once"},
                    ],
                }})
                continue
            _answer(rid, sid, prompt)
        elif rid == 900 and pending_perm is not None:
            prid, sid, prompt = pending_perm
            pending_perm = None
            outcome = (msg.get("result") or {}).get("outcome") or {}
            allowed = outcome.get("outcome") == "selected" and outcome.get("optionId") == "a1"
            notify(sid, {"sessionUpdate": "tool_call", "toolCallId": "tc1", "title": "Run shell: ls",
                         "kind": "execute", "status": "completed" if allowed else "failed"})
            notify(sid, {"sessionUpdate": "agent_message_chunk",
                         "content": {"type": "text", "text": "TOOL RAN" if allowed else "DENIED"}})
            send({"jsonrpc": "2.0", "id": prid, "result": {"stopReason": "end_turn"}})
        elif method == "session/cancel":
            pass
        else:
            send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"unknown {method}"}})


def _answer(rid, sid, prompt):
    text = json.dumps(GOOD) if "Output contract" in prompt else PLAIN
    half = len(text) // 2
    notify(sid, {"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "thinking"}})
    notify(sid, {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text[:half]}})
    notify(sid, {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text[half:]}})
    send({"jsonrpc": "2.0", "id": rid, "result": {
        "stopReason": "end_turn",
        "usage": {"inputTokens": 21, "outputTokens": 6},
    }})


if __name__ == "__main__":
    main()
