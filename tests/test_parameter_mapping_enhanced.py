#!/usr/bin/env python3
"""
Unit tests for enhanced OpenAI -> Claude Code SDK parameter mapping.

Covers the best-effort handling implemented in
``ChatCompletionRequest.get_sampling_instructions()``,
``ChatCompletionRequest.log_parameter_info()``,
``ChatCompletionRequest.to_claude_options()`` and the
``CompatibilityReporter`` classification (best-effort vs unsupported).

These are pure unit tests that do not require a running server.
Run with: python -m pytest tests/test_parameter_mapping_enhanced.py -q
"""

import pytest
from unittest.mock import patch

from src.models import Message, ChatCompletionRequest
from src.parameter_validator import CompatibilityReporter


def _request(**kwargs) -> ChatCompletionRequest:
    """Build a minimal request, overriding fields via kwargs."""
    params = {
        "model": "claude-sonnet-4-5-20250929",
        "messages": [Message(role="user", content="Hello")],
    }
    params.update(kwargs)
    return ChatCompletionRequest(**params)


class TestSamplingInstructionsTemperature:
    """get_sampling_instructions() temperature tiers."""

    def test_default_temperature_returns_none(self):
        """temperature=1.0 (default) produces no instructions."""
        assert _request(temperature=1.0).get_sampling_instructions() is None

    def test_very_low_temperature_is_deterministic(self):
        """temperature < 0.3 asks for deterministic output."""
        instr = _request(temperature=0.1).get_sampling_instructions()
        assert instr is not None
        assert "deterministic" in instr.lower()

    def test_low_mid_temperature_is_focused(self):
        """0.3 <= temperature < 0.7 asks for focused/consistent output."""
        instr = _request(temperature=0.5).get_sampling_instructions()
        assert instr is not None
        assert "consistent" in instr.lower() or "focused" in instr.lower()

    def test_mid_temperature_returns_none(self):
        """0.7 <= temperature <= 1.0 stays neutral (no instruction)."""
        assert _request(temperature=0.9).get_sampling_instructions() is None

    def test_high_temperature_is_creative(self):
        """temperature just above 1.0 asks for creative/varied output."""
        instr = _request(temperature=1.2).get_sampling_instructions()
        assert instr is not None
        assert "creative" in instr.lower() or "varied" in instr.lower()

    def test_very_high_temperature_is_highly_creative(self):
        """temperature > 1.5 asks for highly creative/exploratory output."""
        instr = _request(temperature=1.9).get_sampling_instructions()
        assert instr is not None
        assert "creative" in instr.lower()
        assert "exploratory" in instr.lower()


class TestSamplingInstructionsTopP:
    """get_sampling_instructions() top_p tiers."""

    def test_default_top_p_returns_none(self):
        """top_p=1.0 (default) produces no instructions."""
        assert _request(top_p=1.0).get_sampling_instructions() is None

    def test_very_low_top_p_is_mainstream(self):
        """top_p < 0.5 asks for the most probable/mainstream solutions."""
        instr = _request(top_p=0.3).get_sampling_instructions()
        assert instr is not None
        assert "probable" in instr.lower() or "mainstream" in instr.lower()

    def test_mid_top_p_prefers_common(self):
        """0.5 <= top_p < 0.9 prefers well-established approaches."""
        instr = _request(top_p=0.7).get_sampling_instructions()
        assert instr is not None
        assert "well-established" in instr.lower() or "common" in instr.lower()

    def test_high_top_p_returns_none(self):
        """0.9 <= top_p < 1.0 stays neutral (no instruction)."""
        assert _request(top_p=0.95).get_sampling_instructions() is None


class TestSamplingInstructionsPenalties:
    """get_sampling_instructions() presence/frequency penalty guidance."""

    def test_zero_penalties_return_none(self):
        """Default (0) penalties produce no instructions."""
        assert _request(
            presence_penalty=0, frequency_penalty=0
        ).get_sampling_instructions() is None

    def test_positive_presence_penalty_encourages_new_topics(self):
        """Positive presence_penalty encourages new ideas/topics."""
        instr = _request(presence_penalty=0.8).get_sampling_instructions()
        assert instr is not None
        assert "new" in instr.lower()

    def test_negative_presence_penalty_tolerates_revisiting(self):
        """Negative presence_penalty tolerates revisiting ideas."""
        instr = _request(presence_penalty=-0.8).get_sampling_instructions()
        assert instr is not None
        assert "revisit" in instr.lower() or "stay on topic" in instr.lower()

    def test_positive_frequency_penalty_avoids_repetition(self):
        """Positive frequency_penalty asks to vary wording."""
        instr = _request(frequency_penalty=0.8).get_sampling_instructions()
        assert instr is not None
        assert "repeat" in instr.lower() or "vary" in instr.lower()

    def test_negative_frequency_penalty_tolerates_repetition(self):
        """Negative frequency_penalty tolerates repetition."""
        instr = _request(frequency_penalty=-0.8).get_sampling_instructions()
        assert instr is not None
        assert "repeat" in instr.lower()


class TestSamplingInstructionsCombined:
    """Multiple non-default params compose into one string."""

    def test_combined_params_all_present(self):
        """All best-effort sampling params combine into a single string."""
        instr = _request(
            temperature=0.1,
            top_p=0.3,
            presence_penalty=1.0,
            frequency_penalty=1.0,
        ).get_sampling_instructions()
        assert instr is not None
        assert "deterministic" in instr.lower()
        assert "probable" in instr.lower() or "mainstream" in instr.lower()
        assert "new" in instr.lower()
        assert "vary" in instr.lower() or "repeat" in instr.lower()

    def test_return_type_is_optional_str(self):
        """Return contract is a single Optional[str], not a tuple/list."""
        result = _request(temperature=0.1).get_sampling_instructions()
        assert isinstance(result, str)
        assert _request().get_sampling_instructions() is None


class TestLogParameterInfo:
    """log_parameter_info() routes params to info vs warning correctly."""

    def test_penalties_logged_as_info_not_warning(self):
        """Non-zero penalties are now best-effort (info), not warnings."""
        request = _request(presence_penalty=0.5, frequency_penalty=0.5)
        with patch("src.models.logger") as mock_logger:
            request.log_parameter_info()
            assert mock_logger.info.called
            # No warning should be emitted for penalties alone.
            assert not mock_logger.warning.called

    def test_logit_bias_logged_as_warning(self):
        """logit_bias remains an unsupported warning."""
        request = _request(logit_bias={"hello": 2.0})
        with patch("src.models.logger") as mock_logger:
            request.log_parameter_info()
            assert mock_logger.warning.called
            assert "logit_bias" in str(mock_logger.warning.call_args)

    def test_stop_logged_as_warning(self):
        """stop sequences remain an unsupported warning."""
        request = _request(stop=["END"])
        with patch("src.models.logger") as mock_logger:
            request.log_parameter_info()
            assert mock_logger.warning.called
            assert "stop" in str(mock_logger.warning.call_args).lower()

    def test_temperature_and_top_p_logged_as_info(self):
        """temperature/top_p are best-effort info messages."""
        request = _request(temperature=0.5, top_p=0.5)
        with patch("src.models.logger") as mock_logger:
            request.log_parameter_info()
            assert mock_logger.info.called
            assert not mock_logger.warning.called


class TestToClaudeOptionsMaxTokens:
    """to_claude_options() max_tokens mapping."""

    def test_max_tokens_maps_to_max_thinking_tokens(self):
        """max_tokens is mapped to max_thinking_tokens (approximate)."""
        options = _request(max_tokens=500).to_claude_options()
        assert options.get("max_thinking_tokens") == 500

    def test_max_completion_tokens_takes_precedence(self):
        """max_completion_tokens wins over max_tokens."""
        options = _request(max_tokens=500, max_completion_tokens=1000).to_claude_options()
        assert options.get("max_thinking_tokens") == 1000

    def test_no_max_tokens_omits_thinking_tokens(self):
        """Without max_tokens, max_thinking_tokens is not set."""
        options = _request().to_claude_options()
        assert "max_thinking_tokens" not in options

    def test_sampling_params_not_in_options(self):
        """temperature/top_p/penalties are not placed in the options dict."""
        options = _request(
            temperature=0.2, top_p=0.3, presence_penalty=1.0, frequency_penalty=1.0
        ).to_claude_options()
        assert "temperature" not in options
        assert "top_p" not in options
        assert "presence_penalty" not in options
        assert "frequency_penalty" not in options


class TestCompatibilityReporterClassification:
    """CompatibilityReporter best-effort vs unsupported buckets."""

    def test_report_has_best_effort_section(self):
        """Report exposes a best_effort_parameters section."""
        report = CompatibilityReporter.generate_compatibility_report(_request())
        assert "best_effort_parameters" in report

    def test_temperature_is_best_effort(self):
        report = CompatibilityReporter.generate_compatibility_report(
            _request(temperature=0.5)
        )
        assert "temperature" in report["best_effort_parameters"]
        assert "temperature" not in report["unsupported_parameters"]

    def test_top_p_is_best_effort(self):
        report = CompatibilityReporter.generate_compatibility_report(_request(top_p=0.5))
        assert "top_p" in report["best_effort_parameters"]
        assert "top_p" not in report["unsupported_parameters"]

    def test_penalties_are_best_effort(self):
        report = CompatibilityReporter.generate_compatibility_report(
            _request(presence_penalty=0.5, frequency_penalty=0.5)
        )
        assert "presence_penalty" in report["best_effort_parameters"]
        assert "frequency_penalty" in report["best_effort_parameters"]

    def test_max_tokens_is_best_effort(self):
        report = CompatibilityReporter.generate_compatibility_report(
            _request(max_tokens=200)
        )
        assert "max_tokens" in report["best_effort_parameters"]
        assert "max_tokens" not in report["unsupported_parameters"]

    def test_logit_bias_is_unsupported(self):
        report = CompatibilityReporter.generate_compatibility_report(
            _request(logit_bias={"hi": 2.0})
        )
        assert "logit_bias" in report["unsupported_parameters"]
        assert "logit_bias" not in report["best_effort_parameters"]

    def test_stop_is_unsupported(self):
        report = CompatibilityReporter.generate_compatibility_report(
            _request(stop=["END"])
        )
        assert "stop" in report["unsupported_parameters"]
        assert "stop" not in report["best_effort_parameters"]

    def test_minimal_request_has_no_best_effort_or_unsupported(self):
        report = CompatibilityReporter.generate_compatibility_report(_request())
        assert len(report["best_effort_parameters"]) == 0
        assert len(report["unsupported_parameters"]) == 0
