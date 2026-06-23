#!/usr/bin/env python3
"""
Tests for enhanced streaming chunk handling.

Two layers:

1. Pure unit tests for the new ``MessageAdapter`` streaming helpers
   (``filter_content_streaming`` and ``segment_text``).
2. Endpoint-level SSE tests driven by FastAPI's ``TestClient`` with a
   monkeypatched ``claude_cli.run_completion`` so we control the exact
   sequence of fake SDK chunks and can assert the resulting wire format.
"""

import json
from typing import Any, AsyncGenerator, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

import src.main as main
from src.message_adapter import MessageAdapter


# ---------------------------------------------------------------------------
# Unit tests: MessageAdapter.filter_content_streaming
# ---------------------------------------------------------------------------


class TestFilterContentStreaming:
    def test_empty_returns_empty_not_fallback(self):
        # Crucially, unlike filter_content, the streaming variant must NOT
        # substitute placeholder text for empty / whitespace input.
        assert MessageAdapter.filter_content_streaming("") == ""
        assert MessageAdapter.filter_content_streaming("   ").isspace()
        assert MessageAdapter.filter_content_streaming("\n\n") == "\n\n"

    def test_preserves_inter_token_whitespace(self):
        # Leading/trailing whitespace within a delta is preserved (not stripped),
        # so concatenating deltas reproduces the original text.
        assert MessageAdapter.filter_content_streaming(" world") == " world"
        assert MessageAdapter.filter_content_streaming("Hello ") == "Hello "

    def test_strips_thinking_blocks(self):
        out = MessageAdapter.filter_content_streaming("<thinking>secret</thinking>answer")
        assert "secret" not in out
        assert "answer" in out

    def test_strips_tool_blocks(self):
        out = MessageAdapter.filter_content_streaming("<bash>ls -la</bash>done")
        assert "ls -la" not in out
        assert "done" in out

    def test_passes_through_plain_text(self):
        assert MessageAdapter.filter_content_streaming("just text") == "just text"


# ---------------------------------------------------------------------------
# Unit tests: MessageAdapter.segment_text
# ---------------------------------------------------------------------------


class TestSegmentText:
    def test_no_segmentation_when_disabled(self):
        assert MessageAdapter.segment_text("hello world", 0) == ["hello world"]

    def test_empty_text_yields_no_segments(self):
        assert MessageAdapter.segment_text("", 10) == []

    def test_short_text_single_segment(self):
        assert MessageAdapter.segment_text("hi", 10) == ["hi"]

    def test_segments_reassemble_to_original(self):
        text = "The quick brown fox jumps over the lazy dog"
        segments = MessageAdapter.segment_text(text, 10)
        assert len(segments) > 1
        assert "".join(segments) == text
        assert all(s for s in segments)  # no empty segments

    def test_respects_max_size_bound(self):
        text = "abcdefghijklmnopqrstuvwxyz" * 3  # no whitespace -> hard splits
        segments = MessageAdapter.segment_text(text, 10)
        assert "".join(segments) == text
        assert all(len(s) <= 10 for s in segments)


# ---------------------------------------------------------------------------
# Endpoint-level streaming tests
# ---------------------------------------------------------------------------


def _make_text_block(text: str):
    """A minimal stand-in for the SDK's TextBlock (duck-typed via .text)."""

    class _TextBlock:
        def __init__(self, t: str):
            self.text = t

    return _TextBlock(text)


def _assistant_chunk(text: str) -> Dict[str, Any]:
    """An incremental AssistantMessage chunk (new SDK shape)."""
    return {"content": [_make_text_block(text)], "model": "claude-sonnet-test"}


def _result_chunk(full_text: str) -> Dict[str, Any]:
    """A terminating ResultMessage chunk carrying usage + the full text."""
    return {
        "subtype": "success",
        "result": full_text,
        "total_cost_usd": 0.0001,
        "duration_ms": 42,
        "num_turns": 1,
        "session_id": "sess-test",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _error_chunk(message: str) -> Dict[str, Any]:
    return {
        "type": "result",
        "subtype": "error_during_execution",
        "is_error": True,
        "error_message": message,
    }


def _install_fake_completion(monkeypatch, chunks: List[Dict[str, Any]]):
    """Patch claude_cli.run_completion to yield the supplied fake chunks."""

    async def fake_run_completion(*args, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        for c in chunks:
            yield c

    monkeypatch.setattr(main.claude_cli, "run_completion", fake_run_completion)
    # Bypass Claude Code auth validation so the endpoint reaches the stream path.
    monkeypatch.setattr(main, "validate_claude_code_auth", lambda: (True, {"method": "test"}))


def _parse_sse(body: str) -> List[Any]:
    """Parse an SSE body into a list of payloads ([DONE] kept as a string)."""
    events: List[Any] = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            events.append("[DONE]")
        else:
            events.append(json.loads(payload))
    return events


@pytest.fixture
def client():
    return TestClient(main.app)


def _post_stream(client, monkeypatch, chunks, *, include_usage=False):
    _install_fake_completion(monkeypatch, chunks)
    body: Dict[str, Any] = {
        "model": "claude-sonnet-test",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }
    if include_usage:
        body["stream_options"] = {"include_usage": True}
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    return _parse_sse(resp.text)


class TestStreamingEndpoint:
    def test_basic_stream_shape(self, client, monkeypatch):
        events = _post_stream(
            client,
            monkeypatch,
            [
                _assistant_chunk("Hello"),
                _assistant_chunk(", world"),
                _result_chunk("Hello, world"),
            ],
        )

        # Last event must be the [DONE] sentinel.
        assert events[-1] == "[DONE]"

        # First data event is the role delta.
        first = events[0]
        assert first["object"] == "chat.completion.chunk"
        assert first["choices"][0]["delta"].get("role") == "assistant"

        # Content deltas: non-empty, and not duplicated by the result message.
        content_pieces = [
            e["choices"][0]["delta"].get("content")
            for e in events
            if isinstance(e, dict)
            and e["choices"][0]["delta"].get("content")
            and e["choices"][0]["finish_reason"] is None
        ]
        assert content_pieces == ["Hello", ", world"]
        assert all(c and not c.isspace() for c in content_pieces)

        # Final (pre-DONE) chunk carries finish_reason == "stop".
        final = events[-2]
        assert final["choices"][0]["finish_reason"] == "stop"
        assert final["choices"][0]["delta"] == {}

    def test_no_empty_content_deltas(self, client, monkeypatch):
        events = _post_stream(
            client,
            monkeypatch,
            [
                _assistant_chunk("real"),
                _assistant_chunk("   "),  # whitespace-only -> must be skipped
                _assistant_chunk(""),  # empty -> must be skipped
                _result_chunk("real"),
            ],
        )
        contents = [
            e["choices"][0]["delta"].get("content")
            for e in events
            if isinstance(e, dict) and e["choices"][0]["finish_reason"] is None
        ]
        # Only the role delta ("") and the single "real" delta.
        assert "real" in contents
        # No whitespace-only or stray placeholder content deltas after role.
        non_role = [
            e
            for e in events
            if isinstance(e, dict)
            and e["choices"][0]["delta"].get("role") is None
            and e["choices"][0]["finish_reason"] is None
        ]
        for e in non_role:
            c = e["choices"][0]["delta"].get("content")
            assert c and not c.isspace()

    def test_result_message_not_duplicated(self, client, monkeypatch):
        # The result text equals the concatenated assistant deltas; it must not
        # be streamed a second time.
        events = _post_stream(
            client,
            monkeypatch,
            [_assistant_chunk("ABC"), _result_chunk("ABC")],
        )
        contents = [
            e["choices"][0]["delta"].get("content")
            for e in events
            if isinstance(e, dict)
            and e["choices"][0]["delta"].get("content")
            and e["choices"][0]["finish_reason"] is None
        ]
        assert contents == ["ABC"]

    def test_usage_chunk_when_requested(self, client, monkeypatch):
        events = _post_stream(
            client,
            monkeypatch,
            [_assistant_chunk("hi"), _result_chunk("hi")],
            include_usage=True,
        )
        final = events[-2]
        assert final["choices"][0]["finish_reason"] == "stop"
        assert final.get("usage") is not None
        usage = final["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_no_usage_chunk_by_default(self, client, monkeypatch):
        events = _post_stream(
            client,
            monkeypatch,
            [_assistant_chunk("hi"), _result_chunk("hi")],
        )
        final = events[-2]
        assert final.get("usage") is None

    def test_mid_stream_error_terminates_cleanly(self, client, monkeypatch):
        events = _post_stream(
            client,
            monkeypatch,
            [_error_chunk("boom")],
        )
        # Even on error we produce a well-formed stream: role, a content delta
        # describing the error, a stop chunk, and [DONE].
        assert events[-1] == "[DONE]"
        assert events[0]["choices"][0]["delta"].get("role") == "assistant"
        assert events[-2]["choices"][0]["finish_reason"] == "stop"
        error_contents = [
            e["choices"][0]["delta"].get("content")
            for e in events
            if isinstance(e, dict) and e["choices"][0]["delta"].get("content")
        ]
        assert any("boom" in (c or "") for c in error_contents)

    def test_error_after_content_still_finishes(self, client, monkeypatch):
        events = _post_stream(
            client,
            monkeypatch,
            [_assistant_chunk("partial"), _error_chunk("late failure")],
        )
        assert events[-1] == "[DONE]"
        assert events[-2]["choices"][0]["finish_reason"] == "stop"
        contents = [
            e["choices"][0]["delta"].get("content")
            for e in events
            if isinstance(e, dict)
            and e["choices"][0]["delta"].get("content")
            and e["choices"][0]["finish_reason"] is None
        ]
        assert "partial" in contents
