#!/usr/bin/env python3
"""
Unit and endpoint tests for prompt-based OpenAI function/tool calling.

Covers:
- System-prompt fragment construction (build_tool_prompt)
- Envelope parsing incl. malformed / no-call cases (parse_tool_calls)
- Legacy functions/function_call normalization
- id / arguments-string output shape
- An endpoint-level test using FastAPI TestClient with claude_cli.run_completion
  monkeypatched to return a canned tool-call envelope.
"""

import json

import pytest

from src.function_calling import (
    ENVELOPE_KEY,
    build_tool_prompt,
    normalize_legacy_functions,
    parse_tool_calls,
    resolve_tools,
)


# ---------------------------------------------------------------------------
# Fixtures / sample data
# ---------------------------------------------------------------------------

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]


def _envelope_text(calls):
    """Build a fenced json envelope string from a list of {name, arguments} dicts."""
    body = json.dumps({ENVELOPE_KEY: calls}, indent=2)
    return f"Here you go:\n```json\n{body}\n```"


# ---------------------------------------------------------------------------
# build_tool_prompt
# ---------------------------------------------------------------------------


class TestBuildToolPrompt:
    def test_returns_none_without_tools(self):
        assert build_tool_prompt(None) is None
        assert build_tool_prompt([]) is None

    def test_includes_function_name_and_envelope_key(self):
        prompt = build_tool_prompt(SAMPLE_TOOLS)
        assert prompt is not None
        assert "get_weather" in prompt
        assert ENVELOPE_KEY in prompt
        # Should mention a json fenced block.
        assert "```json" in prompt

    def test_includes_parameters_schema(self):
        prompt = build_tool_prompt(SAMPLE_TOOLS)
        assert "location" in prompt
        assert "fahrenheit" in prompt

    def test_tool_choice_none_disables(self):
        assert build_tool_prompt(SAMPLE_TOOLS, "none") is None

    def test_tool_choice_required_forces_call(self):
        prompt = build_tool_prompt(SAMPLE_TOOLS, "required")
        assert prompt is not None
        assert "MUST call" in prompt

    def test_tool_choice_specific_function(self):
        choice = {"type": "function", "function": {"name": "get_weather"}}
        prompt = build_tool_prompt(SAMPLE_TOOLS, choice)
        assert prompt is not None
        assert "get_weather" in prompt
        assert "MUST call" in prompt

    def test_tool_choice_auto_is_conditional(self):
        prompt = build_tool_prompt(SAMPLE_TOOLS, "auto")
        assert prompt is not None
        assert "only if" in prompt.lower()

    def test_malformed_tool_skipped(self):
        # A tool missing the function name yields no usable tools -> None.
        assert build_tool_prompt([{"type": "function", "function": {}}]) is None
        assert build_tool_prompt([{"not": "a tool"}]) is None


# ---------------------------------------------------------------------------
# parse_tool_calls
# ---------------------------------------------------------------------------


class TestParseToolCalls:
    def test_none_and_empty(self):
        assert parse_tool_calls(None) is None
        assert parse_tool_calls("") is None

    def test_plain_text_returns_none(self):
        assert parse_tool_calls("The weather is sunny today.") is None

    def test_parses_single_call(self):
        text = _envelope_text([{"name": "get_weather", "arguments": {"location": "Paris"}}])
        calls = parse_tool_calls(text)
        assert calls is not None
        assert len(calls) == 1
        call = calls[0]
        assert call["type"] == "function"
        assert call["id"].startswith("call_")
        assert call["function"]["name"] == "get_weather"
        # arguments must be a JSON *string*.
        assert isinstance(call["function"]["arguments"], str)
        assert json.loads(call["function"]["arguments"]) == {"location": "Paris"}

    def test_parses_multiple_calls(self):
        text = _envelope_text(
            [
                {"name": "get_weather", "arguments": {"location": "Paris"}},
                {"name": "get_weather", "arguments": {"location": "Tokyo"}},
            ]
        )
        calls = parse_tool_calls(text)
        assert calls is not None
        assert len(calls) == 2
        # Each call gets a unique id.
        assert calls[0]["id"] != calls[1]["id"]

    def test_arguments_already_string_passthrough(self):
        text = _envelope_text([{"name": "get_weather", "arguments": '{"location": "Paris"}'}])
        calls = parse_tool_calls(text)
        assert calls is not None
        assert calls[0]["function"]["arguments"] == '{"location": "Paris"}'

    def test_missing_arguments_defaults_to_empty_object(self):
        text = _envelope_text([{"name": "do_thing"}])
        calls = parse_tool_calls(text)
        assert calls is not None
        assert json.loads(calls[0]["function"]["arguments"]) == {}

    def test_bare_json_object_without_fence(self):
        body = json.dumps(
            {ENVELOPE_KEY: [{"name": "get_weather", "arguments": {"location": "Rome"}}]}
        )
        calls = parse_tool_calls(body)
        assert calls is not None
        assert calls[0]["function"]["name"] == "get_weather"

    def test_malformed_json_returns_none(self):
        text = "```json\n{ this is not valid json }\n```"
        assert parse_tool_calls(text) is None

    def test_envelope_without_tool_calls_key_returns_none(self):
        text = '```json\n{"result": 42}\n```'
        assert parse_tool_calls(text) is None

    def test_call_without_name_skipped(self):
        text = _envelope_text([{"arguments": {"x": 1}}])
        assert parse_tool_calls(text) is None

    def test_non_list_tool_calls_returns_none(self):
        text = '```json\n{"tool_calls": "nope"}\n```'
        assert parse_tool_calls(text) is None


# ---------------------------------------------------------------------------
# Legacy normalization
# ---------------------------------------------------------------------------


class TestLegacyNormalization:
    def test_normalize_functions_to_tools(self):
        functions = [{"name": "f", "description": "d", "parameters": {"type": "object"}}]
        tools, choice = normalize_legacy_functions(functions, None)
        assert tools == [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "description": "d",
                    "parameters": {"type": "object"},
                },
            }
        ]
        assert choice is None

    def test_normalize_function_call_string(self):
        _tools, choice = normalize_legacy_functions(None, "auto")
        assert choice == "auto"

    def test_normalize_function_call_dict(self):
        _tools, choice = normalize_legacy_functions(None, {"name": "f"})
        assert choice == {"type": "function", "function": {"name": "f"}}

    def test_normalize_none(self):
        tools, choice = normalize_legacy_functions(None, None)
        assert tools is None
        assert choice is None

    def test_resolve_prefers_modern_over_legacy(self):
        legacy_functions = [{"name": "old"}]
        tools, _choice = resolve_tools(SAMPLE_TOOLS, None, legacy_functions, None)
        assert tools == SAMPLE_TOOLS

    def test_resolve_falls_back_to_legacy(self):
        legacy_functions = [{"name": "old", "parameters": {}}]
        tools, choice = resolve_tools(None, None, legacy_functions, "auto")
        assert tools[0]["function"]["name"] == "old"
        assert choice == "auto"


# ---------------------------------------------------------------------------
# Endpoint-level test
# ---------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    """A TestClient with auth and Claude SDK calls stubbed out."""
    from fastapi.testclient import TestClient
    import src.main as main

    # Bypass Claude Code auth validation.
    monkeypatch.setattr(main, "validate_claude_code_auth", lambda: (True, {"method": "test"}))

    # Stub the structured logging helpers (avoid real cost/token computation).
    monkeypatch.setattr(main, "_log_claude_proxy_start", lambda *a, **k: None)
    monkeypatch.setattr(main, "_log_claude_proxy_success", lambda *a, **k: (10, 5, 0.0))

    return TestClient(main.app)


def _make_run_completion(result_text):
    """Build an async-generator stand-in for claude_cli.run_completion."""

    async def fake_run_completion(*args, **kwargs):
        yield {"subtype": "success", "result": result_text}

    return fake_run_completion


def test_endpoint_returns_tool_calls(client, monkeypatch):
    import src.main as main

    envelope = _envelope_text([{"name": "get_weather", "arguments": {"location": "Paris"}}])
    monkeypatch.setattr(main.claude_cli, "run_completion", _make_run_completion(envelope))

    payload = {
        "model": "claude-3-5-haiku-20241022",
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": SAMPLE_TOOLS,
        "tool_choice": "auto",
        "stream": False,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()

    choice = data["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    message = choice["message"]
    assert message["content"] is None
    tool_calls = message["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["id"].startswith("call_")
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"location": "Paris"}


def test_endpoint_plain_text_when_no_tool_call(client, monkeypatch):
    """With tools present but a plain-text response, normal text path is used."""
    import src.main as main

    monkeypatch.setattr(
        main.claude_cli,
        "run_completion",
        _make_run_completion("It is sunny in Paris."),
    )

    payload = {
        "model": "claude-3-5-haiku-20241022",
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": SAMPLE_TOOLS,
        "stream": False,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "It is sunny in Paris."
    assert choice["message"]["tool_calls"] is None


def test_endpoint_no_tools_unchanged(client, monkeypatch):
    """Without tools, behavior is the normal text path (regression guard)."""
    import src.main as main

    monkeypatch.setattr(main.claude_cli, "run_completion", _make_run_completion("Hello there."))

    payload = {
        "model": "claude-3-5-haiku-20241022",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "Hello there."
