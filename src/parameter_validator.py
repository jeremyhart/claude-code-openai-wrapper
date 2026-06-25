"""
Parameter validation and mapping utilities for OpenAI to Claude Code SDK conversion.
"""

import logging
from typing import Dict, Any, List, Optional
from src.models import ChatCompletionRequest
from src.constants import CLAUDE_MODELS

logger = logging.getLogger(__name__)


class ParameterValidator:
    """Validates and maps OpenAI Chat Completions parameters to Claude Code SDK options."""

    # Use models from constants (single source of truth)
    SUPPORTED_MODELS = set(CLAUDE_MODELS)

    # Valid permission modes for Claude Code SDK
    VALID_PERMISSION_MODES = {"default", "acceptEdits", "bypassPermissions", "plan"}

    @classmethod
    def validate_model(cls, model: str) -> bool:
        """Validate that the model is supported by Claude Code SDK."""
        if model not in cls.SUPPORTED_MODELS:
            logger.warning(
                f"Model '{model}' is not in the known supported models list. It will still be attempted but may fail. Supported models: {sorted(cls.SUPPORTED_MODELS)}"
            )
            # Return True anyway to allow graceful degradation
        return True

    @classmethod
    def validate_permission_mode(cls, permission_mode: str) -> bool:
        """Validate permission mode parameter."""
        if permission_mode not in cls.VALID_PERMISSION_MODES:
            logger.error(
                f"Invalid permission_mode '{permission_mode}'. Valid options: {cls.VALID_PERMISSION_MODES}"
            )
            return False
        return True

    @classmethod
    def validate_tools(cls, tools: List[str]) -> bool:
        """Validate tool names (basic validation for non-empty strings)."""
        if not all(isinstance(tool, str) and tool.strip() for tool in tools):
            logger.error("All tool names must be non-empty strings")
            return False
        return True

    @classmethod
    def create_enhanced_options(
        cls,
        request: ChatCompletionRequest,
        max_turns: Optional[int] = None,
        allowed_tools: Optional[List[str]] = None,
        disallowed_tools: Optional[List[str]] = None,
        permission_mode: Optional[str] = None,
        max_thinking_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Create enhanced Claude Code SDK options with additional parameters.

        This allows API users to pass Claude-Code-specific parameters that don't
        exist in the OpenAI API through custom headers or environment variables.
        """
        # Start with basic options from request
        options = request.to_claude_options()

        # Add Claude Code SDK specific options
        if max_turns is not None:
            if max_turns < 1 or max_turns > 100:
                logger.warning(f"max_turns={max_turns} is outside recommended range (1-100)")
            options["max_turns"] = max_turns

        if allowed_tools:
            if cls.validate_tools(allowed_tools):
                options["allowed_tools"] = allowed_tools

        if disallowed_tools:
            if cls.validate_tools(disallowed_tools):
                options["disallowed_tools"] = disallowed_tools

        if permission_mode:
            if cls.validate_permission_mode(permission_mode):
                options["permission_mode"] = permission_mode

        if max_thinking_tokens is not None:
            if max_thinking_tokens < 0 or max_thinking_tokens > 50000:
                logger.warning(
                    f"max_thinking_tokens={max_thinking_tokens} is outside recommended range (0-50000)"
                )
            options["max_thinking_tokens"] = max_thinking_tokens

        return options

    @classmethod
    def extract_claude_headers(cls, headers: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract Claude-Code-specific parameters from custom HTTP headers.

        This allows clients to pass SDK-specific options via headers:
        - X-Claude-Max-Turns: 5
        - X-Claude-Allowed-Tools: tool1,tool2,tool3
        - X-Claude-Permission-Mode: acceptEdits
        """
        claude_options = {}

        # Extract max_turns
        if "x-claude-max-turns" in headers:
            try:
                claude_options["max_turns"] = int(headers["x-claude-max-turns"])
            except ValueError:
                logger.warning(
                    f"Invalid X-Claude-Max-Turns header: {headers['x-claude-max-turns']}"
                )

        # Extract allowed tools
        if "x-claude-allowed-tools" in headers:
            tools = [tool.strip() for tool in headers["x-claude-allowed-tools"].split(",")]
            if tools:
                claude_options["allowed_tools"] = tools

        # Extract disallowed tools
        if "x-claude-disallowed-tools" in headers:
            tools = [tool.strip() for tool in headers["x-claude-disallowed-tools"].split(",")]
            if tools:
                claude_options["disallowed_tools"] = tools

        # Extract permission mode
        if "x-claude-permission-mode" in headers:
            claude_options["permission_mode"] = headers["x-claude-permission-mode"]

        # Extract max thinking tokens
        if "x-claude-max-thinking-tokens" in headers:
            try:
                claude_options["max_thinking_tokens"] = int(headers["x-claude-max-thinking-tokens"])
            except ValueError:
                logger.warning(
                    f"Invalid X-Claude-Max-Thinking-Tokens header: {headers['x-claude-max-thinking-tokens']}"
                )

        return claude_options


class CompatibilityReporter:
    """Reports on OpenAI API compatibility and suggests alternatives."""

    @classmethod
    def generate_compatibility_report(cls, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Generate a detailed compatibility report for the request.

        Parameters fall into three buckets:

        - ``supported_parameters``: handled natively (model, messages, etc.)
          or enforced by post-processing (stop, via output truncation).
        - ``best_effort_parameters``: no native SDK knob, but approximated via
          system-prompt sampling instructions (temperature, top_p,
          presence_penalty, frequency_penalty) or an approximate SDK mapping
          (max_tokens -> max_thinking_tokens).
        - ``unsupported_parameters``: genuinely ignored (logit_bias).

        ``best_effort_parameters`` are intentionally NOT added to
        ``unsupported_parameters`` so callers can distinguish "approximated"
        from "ignored".
        """
        report = {
            "supported_parameters": [],
            "best_effort_parameters": [],
            "unsupported_parameters": [],
            "warnings": [],
            "suggestions": [],
        }

        # Check supported parameters
        if request.model:
            report["supported_parameters"].append("model")
        if request.messages:
            report["supported_parameters"].append("messages")
        if request.stream is not None:
            report["supported_parameters"].append("stream")
        if request.user:
            report["supported_parameters"].append("user (for logging)")

        # Best-effort parameters: approximated via system-prompt instructions.
        if request.temperature is not None and request.temperature != 1.0:
            report["best_effort_parameters"].append("temperature")
            report["suggestions"].append(
                "temperature is approximated via system-prompt sampling instructions rather than native sampling. The effect is qualitative, not exact."
            )

        if request.top_p is not None and request.top_p != 1.0:
            report["best_effort_parameters"].append("top_p")
            report["suggestions"].append(
                "top_p is approximated via system-prompt sampling instructions rather than native nucleus sampling. The effect is qualitative, not exact."
            )

        if request.presence_penalty is not None and request.presence_penalty != 0:
            report["best_effort_parameters"].append("presence_penalty")
            report["suggestions"].append(
                "presence_penalty is approximated via system-prompt guidance encouraging (or tolerating) new topics. The effect is qualitative, not exact."
            )

        if request.frequency_penalty is not None and request.frequency_penalty != 0:
            report["best_effort_parameters"].append("frequency_penalty")
            report["suggestions"].append(
                "frequency_penalty is approximated via system-prompt guidance about repeating words and phrases. The effect is qualitative, not exact."
            )

        if request.max_tokens or request.max_completion_tokens:
            report["best_effort_parameters"].append("max_tokens")
            report["suggestions"].append(
                "max_tokens is approximately mapped to max_thinking_tokens, which bounds internal reasoning rather than visible output length."
            )

        # Multiple choices (n > 1) are supported by running the completion
        # multiple times.
        if request.n is not None and request.n > 1:
            report["supported_parameters"].append("n")
            report["suggestions"].append(
                f"n={request.n} is supported; the completion is run {request.n} times to produce that many choices."
            )

        # Genuinely unsupported parameters.
        if request.stop:
            report["supported_parameters"].append("stop")
            report["suggestions"].append(
                "stop sequences are supported; the response is truncated at the earliest stop string (excluded from the output, OpenAI-style)."
            )

        if request.logit_bias:
            report["unsupported_parameters"].append("logit_bias")
            report["suggestions"].append(
                "Logit bias is not supported. Consider using system prompts to guide response style."
            )

        return report
