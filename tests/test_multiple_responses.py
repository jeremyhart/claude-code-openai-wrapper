#!/usr/bin/env python3
"""
Tests for multiple-response support (n > 1) in the OpenAI-compatible wrapper.

Covers:
- The ChatCompletionRequest.validate_n validator (n=1 ok, n=3 ok, n=0 rejected,
  n > cap rejected).
- The non-streaming /v1/chat/completions endpoint returning n choices with the
  correct indices and aggregated usage, with claude_cli.run_completion mocked.
"""

import os

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

os.environ["DEBUG_MODE"] = "true"

import src.main as main
from src.main import app
from src.models import ChatCompletionRequest


# ---------------------------------------------------------------------------
# Validator unit tests
# ---------------------------------------------------------------------------


def _make_request(n):
    return ChatCompletionRequest(
        model="claude-3-5-haiku-20241022",
        messages=[{"role": "user", "content": "hi"}],
        n=n,
    )


def test_validate_n_accepts_one():
    assert _make_request(1).n == 1


def test_validate_n_accepts_three():
    assert _make_request(3).n == 3


def test_validate_n_rejects_zero():
    with pytest.raises(ValidationError):
        _make_request(0)


def test_validate_n_rejects_above_cap():
    with pytest.raises(ValidationError):
        _make_request(129)


def test_validate_n_accepts_cap():
    assert _make_request(128).n == 128


# ---------------------------------------------------------------------------
# Endpoint behavior tests
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_main(monkeypatch):
    """Patch auth + the Claude CLI so the endpoint runs without a live SDK."""

    # Bypass authentication.
    monkeypatch.setattr(main, "validate_claude_code_auth", lambda: (True, {"method": "test"}))

    async def _noop_verify(request, credentials):
        return None

    monkeypatch.setattr(main, "verify_api_key", _noop_verify)

    # Track how many times run_completion is invoked so each "choice" gets
    # distinct content.
    call_counter = {"count": 0}

    async def fake_run_completion(*args, **kwargs):
        idx = call_counter["count"]
        call_counter["count"] += 1
        # Yield a single assistant chunk in the new content-list format.
        yield {"content": [{"type": "text", "text": f"response-{idx}"}]}

    monkeypatch.setattr(main.claude_cli, "run_completion", fake_run_completion)

    # Deterministic parse/metadata/usage so usage aggregation is predictable.
    def fake_parse(chunks):
        for chunk in chunks:
            for block in chunk.get("content", []):
                if block.get("type") == "text":
                    return block["text"]
        return None

    monkeypatch.setattr(main.claude_cli, "parse_claude_message", fake_parse)
    monkeypatch.setattr(main.claude_cli, "extract_metadata", lambda chunks: {})

    # _log_claude_proxy_success returns (prompt_tokens, completion_tokens, cost).
    monkeypatch.setattr(main, "_log_claude_proxy_success", lambda **kwargs: (5, 7, 0.0))
    monkeypatch.setattr(main, "_log_claude_proxy_start", lambda *a, **k: None)

    return main


def test_non_streaming_n_returns_three_choices(patched_main):
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-3-5-haiku-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "n": 3,
            "stream": False,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    choices = data["choices"]
    assert len(choices) == 3
    assert [c["index"] for c in choices] == [0, 1, 2]
    # Each choice came from a distinct run_completion invocation.
    contents = [c["message"]["content"] for c in choices]
    assert contents == ["response-0", "response-1", "response-2"]
    for c in choices:
        assert c["finish_reason"] == "stop"
        assert c["message"]["role"] == "assistant"

    # Usage aggregated across the 3 runs (5 + 7 each).
    usage = data["usage"]
    assert usage["prompt_tokens"] == 15
    assert usage["completion_tokens"] == 21
    assert usage["total_tokens"] == 36


def test_non_streaming_default_n_one(patched_main):
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-3-5-haiku-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["choices"]) == 1
    assert data["choices"][0]["index"] == 0
    assert data["usage"]["prompt_tokens"] == 5
    assert data["usage"]["completion_tokens"] == 7


def test_streaming_n_emits_per_index_chunks(patched_main):
    client = TestClient(app)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "claude-3-5-haiku-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "n": 2,
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())

    import json

    finish_indices = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :].strip()
        if payload == "[DONE]":
            continue
        obj = json.loads(payload)
        for choice in obj.get("choices", []):
            if choice.get("finish_reason") == "stop":
                finish_indices.append(choice["index"])

    # One finish chunk per choice, with the correct indices.
    assert finish_indices == [0, 1]
    assert body.rstrip().endswith("data: [DONE]")
