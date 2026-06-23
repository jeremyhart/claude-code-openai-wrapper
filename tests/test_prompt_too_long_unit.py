#!/usr/bin/env python3
"""
Unit tests for surfacing upstream Claude/Anthropic API errors as proper,
OpenAI-compatible error responses.

The motivating bug: when a request exceeds the model's context window the CLI
returns a ResultMessage with ``is_error=True`` and ``result="Prompt is too
long"`` (but ``subtype="success"``). The non-streaming paths used to treat that
as a successful completion and hand the raw error string back to the client as
the assistant reply. These tests pin the corrected behavior: a 400 with a
``context_length_exceeded`` error envelope, and accurate error text on the
streaming paths.
"""

import json
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

import src.main as main
from src.claude_cli import classify_sdk_error, find_sdk_error


# --- Unit tests for the pure classification helpers ------------------------


class TestClassifySdkError:
    def test_prompt_too_long_maps_to_context_length_exceeded(self):
        result = classify_sdk_error("Prompt is too long")
        assert result["status_code"] == 400
        assert result["type"] == "invalid_request_error"
        assert result["code"] == "context_length_exceeded"
        assert result["message"] == "Prompt is too long"

    def test_match_is_case_insensitive_substring(self):
        result = classify_sdk_error("API Error: prompt is too long for this model")
        assert result["code"] == "context_length_exceeded"

    def test_credit_balance(self):
        assert classify_sdk_error("Credit balance is too low")["code"] == "insufficient_quota"

    def test_rate_limit(self):
        result = classify_sdk_error("Rate limit exceeded")
        assert result["status_code"] == 429
        assert result["type"] == "rate_limit_error"

    def test_unknown_error_falls_back_to_api_error_status(self):
        result = classify_sdk_error("Some new failure", api_error_status=503)
        assert result["status_code"] == 503
        assert result["type"] == "api_error"

    def test_unknown_4xx_status_is_invalid_request(self):
        result = classify_sdk_error("Bad input", api_error_status=400)
        assert result["type"] == "invalid_request_error"

    def test_unknown_error_no_status_defaults_to_500(self):
        result = classify_sdk_error("Mystery")
        assert result["status_code"] == 500
        assert result["type"] == "api_error"

    def test_empty_message_gets_placeholder(self):
        assert classify_sdk_error("")["message"] == "Claude Code error"


class TestFindSdkError:
    def test_api_error_result_message_detected(self):
        # The CLI emits subtype="success" even for API errors; is_error is the signal.
        chunks = [
            {
                "subtype": "success",
                "is_error": True,
                "result": "Prompt is too long",
                "api_error_status": 400,
            }
        ]
        err = find_sdk_error(chunks)
        assert err is not None
        assert err["code"] == "context_length_exceeded"

    def test_subprocess_error_uses_error_message(self):
        chunks = [
            {"subtype": "error_during_execution", "is_error": True, "error_message": "SDK failed"}
        ]
        err = find_sdk_error(chunks)
        assert err is not None
        assert err["message"] == "SDK failed"

    def test_successful_run_returns_none(self):
        chunks = [
            {"content": [{"type": "text", "text": "hi"}]},
            {"subtype": "success", "is_error": False, "result": "hi"},
        ]
        assert find_sdk_error(chunks) is None


# --- Endpoint tests --------------------------------------------------------


def _error_result_chunk() -> Dict[str, Any]:
    """An oversized-prompt ResultMessage as emitted by the CLI."""
    return {
        "type": "result",
        "subtype": "success",
        "is_error": True,
        "result": "Prompt is too long",
        "api_error_status": 400,
        "session_id": "sess-test",
    }


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(main, "validate_claude_code_auth", lambda: (True, {"method": "test"}))

    async def _noop_verify(request, credentials):
        return None

    monkeypatch.setattr(main, "verify_api_key", _noop_verify)
    monkeypatch.setattr(main, "_log_claude_proxy_start", lambda *a, **k: None)

    chunks: List[Dict[str, Any]] = [_error_result_chunk()]

    async def fake_run_completion(*args, **kwargs):
        for c in chunks:
            yield c

    monkeypatch.setattr(main.claude_cli, "run_completion", fake_run_completion)
    return chunks


def test_chat_completions_non_streaming_returns_400(patched):
    client = TestClient(main.app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "x" * 50}],
            "stream": False,
        },
    )
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["code"] == "context_length_exceeded"
    assert error["type"] == "invalid_request_error"
    assert "Prompt is too long" in error["message"]


def test_messages_non_streaming_returns_400(patched):
    client = TestClient(main.app)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 20,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["code"] == "context_length_exceeded"
    assert "Prompt is too long" in error["message"]


def test_chat_completions_streaming_surfaces_real_message(patched):
    """Streaming can't change status mid-flight, but it must report the real
    reason rather than a generic "Claude Code error"."""
    client = TestClient(main.app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert "Prompt is too long" in resp.text
