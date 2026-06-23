#!/usr/bin/env python3
"""
Unit tests for the /v1/messages endpoint behavior with the Claude SDK mocked.

These exercise the wiring that makes the endpoint behave like the native
Anthropic Messages API on a Claude Code subscription:

  * the caller's ``system`` prompt is forwarded (not silently dropped),
  * the built-in Claude Code tools and ~18k preset prompt are NOT injected,
  * the caller's ``tools`` are honored via prompt-based function calling and
    surface as ``tool_use`` content blocks with ``stop_reason="tool_use"``.
"""

import pytest
from fastapi.testclient import TestClient

import src.main as main
from src.main import app
from src.constants import CLAUDE_TOOLS


@pytest.fixture
def patched(monkeypatch):
    """Patch auth + the Claude CLI so the endpoint runs without a live SDK."""
    monkeypatch.setattr(main, "validate_claude_code_auth", lambda: (True, {"method": "test"}))

    async def _noop_verify(request, credentials):
        return None

    monkeypatch.setattr(main, "verify_api_key", _noop_verify)

    captured = {"kwargs": None}
    # The raw text the fake SDK "returns"; tests override per-case.
    state = {"raw": "Hello there."}

    async def fake_run_completion(*args, **kwargs):
        captured["kwargs"] = kwargs
        yield {"content": [{"type": "text", "text": state["raw"]}]}

    monkeypatch.setattr(main.claude_cli, "run_completion", fake_run_completion)
    monkeypatch.setattr(main.claude_cli, "parse_claude_message", lambda chunks: state["raw"])
    monkeypatch.setattr(main.claude_cli, "extract_metadata", lambda chunks: {})
    monkeypatch.setattr(main, "_log_claude_proxy_success", lambda **kwargs: (12, 7, 0.0))
    monkeypatch.setattr(main, "_log_claude_proxy_start", lambda *a, **k: None)

    return captured, state


def test_system_prompt_forwarded_and_preset_bypassed(patched):
    """Custom system is forwarded; built-in tools + preset are disabled."""
    captured, _ = patched
    client = TestClient(app)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 20,
            "system": "You are Zephyr-7.",
            "messages": [{"role": "user", "content": "name?"}],
        },
    )

    assert resp.status_code == 200
    kwargs = captured["kwargs"]
    assert kwargs["system_prompt"] == "You are Zephyr-7."
    # No Claude Code persona injection, no built-in tools, single turn.
    assert kwargs["use_claude_code_preset"] is False
    assert kwargs["disallowed_tools"] == CLAUDE_TOOLS
    assert kwargs["max_turns"] == 1
    assert kwargs.get("allowed_tools") is None


def test_no_system_still_bypasses_preset(patched):
    """A bare request bypasses the ~18k preset (no hidden persona/bloat)."""
    captured, _ = patched
    client = TestClient(app)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert resp.status_code == 200
    assert captured["kwargs"]["use_claude_code_preset"] is False


def test_tools_are_forwarded_in_prompt(patched):
    """Caller tools are rendered into the system prompt fragment."""
    captured, _ = patched
    client = TestClient(app)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 150,
            "messages": [{"role": "user", "content": "weather in Wellington?"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
        },
    )

    assert resp.status_code == 200
    system_prompt = captured["kwargs"]["system_prompt"]
    assert "get_weather" in system_prompt
    assert "Function calling" in system_prompt


def test_tool_call_envelope_becomes_tool_use_block(patched):
    """A tool-call envelope in the raw response yields a tool_use block."""
    captured, state = patched
    state["raw"] = (
        '```json\n{"tool_calls": [{"name": "get_weather", '
        '"arguments": {"city": "Wellington"}}]}\n```'
    )
    client = TestClient(app)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 150,
            "messages": [{"role": "user", "content": "weather in Wellington? use the tool"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["stop_reason"] == "tool_use"
    types = [block["type"] for block in body["content"]]
    assert "tool_use" in types
    tool_block = next(b for b in body["content"] if b["type"] == "tool_use")
    assert tool_block["name"] == "get_weather"
    assert tool_block["input"] == {"city": "Wellington"}


def test_plain_text_response_when_no_tool_call(patched):
    """Without a tool-call envelope the response is a normal text block."""
    captured, state = patched
    state["raw"] = "It is sunny in Wellington."
    client = TestClient(app)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "weather?"}],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["stop_reason"] == "end_turn"
    assert body["content"][0]["type"] == "text"
    assert body["content"][0]["text"] == "It is sunny in Wellington."


def _parse_sse(text):
    """Parse an Anthropic SSE body into a list of (event, data) tuples."""
    import json

    events = []
    for block in text.strip().split("\n\n"):
        event = None
        data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = json.loads(line[len("data:"):].strip())
        if event is not None:
            events.append((event, data))
    return events


def test_streaming_text_response(patched):
    """A streaming text request emits the native Anthropic event sequence."""
    captured, state = patched
    state["raw"] = "Hello there."
    client = TestClient(app)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 20,
            "system": "You are Zephyr-7.",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(resp.text)
    names = [e for e, _ in events]

    assert names[0] == "message_start"
    assert "content_block_start" in names
    assert "content_block_stop" in names
    assert names[-1] == "message_stop"

    # Text delta carries the streamed content.
    deltas = [d for e, d in events if e == "content_block_delta"]
    streamed = "".join(
        d["delta"]["text"] for d in deltas if d["delta"]["type"] == "text_delta"
    )
    assert "Hello there." in streamed

    # message_delta reports a normal end_turn stop reason.
    msg_delta = next(d for e, d in events if e == "message_delta")
    assert msg_delta["delta"]["stop_reason"] == "end_turn"

    # Streaming still bypasses the preset / built-in tools.
    assert captured["kwargs"]["use_claude_code_preset"] is False
    assert captured["kwargs"]["stream"] is True
    assert captured["kwargs"]["disallowed_tools"] == CLAUDE_TOOLS


def test_streaming_tool_use_response(patched):
    """A streaming tool call emits a tool_use block and stop_reason=tool_use."""
    captured, state = patched
    state["raw"] = (
        '```json\n{"tool_calls": [{"name": "get_weather", '
        '"arguments": {"city": "Wellington"}}]}\n```'
    )
    client = TestClient(app)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 150,
            "stream": True,
            "messages": [{"role": "user", "content": "weather in Wellington? use the tool"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
        },
    )

    assert resp.status_code == 200
    events = _parse_sse(resp.text)

    # A tool_use content block is started.
    starts = [d for e, d in events if e == "content_block_start"]
    assert any(s["content_block"]["type"] == "tool_use" for s in starts)
    tool_start = next(s for s in starts if s["content_block"]["type"] == "tool_use")
    assert tool_start["content_block"]["name"] == "get_weather"

    # The arguments arrive as an input_json_delta.
    deltas = [d for e, d in events if e == "content_block_delta"]
    json_deltas = [d for d in deltas if d["delta"]["type"] == "input_json_delta"]
    assert json_deltas
    import json as _json

    assert _json.loads(json_deltas[0]["delta"]["partial_json"]) == {"city": "Wellington"}

    msg_delta = next(d for e, d in events if e == "message_delta")
    assert msg_delta["delta"]["stop_reason"] == "tool_use"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
