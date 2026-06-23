import os
import tempfile
import atexit
import shutil
from typing import AsyncGenerator, Dict, Any, Optional, List
from pathlib import Path
import logging

from claude_agent_sdk import query, ClaudeAgentOptions

logger = logging.getLogger(__name__)


# The Claude Agent SDK passes the system prompt and the user prompt to the
# `claude` CLI as command-line arguments (``--system-prompt <str>`` and
# ``--print -- <prompt>``). Linux caps a single argv string at ~128KB
# (MAX_ARG_STRLEN); exceeding it makes the subprocess spawn fail with
# ``[Errno 7] Argument list too long``. When a prompt is larger than this
# threshold we route it off the command line instead (system prompt → a temp
# file via ``--append-system-prompt-file``; user prompt → stdin via the SDK's
# streaming-input mode). The threshold is set below the hard limit to leave
# headroom for the flag name, other arguments, and the environment block.
MAX_CLI_ARG_BYTES = 96 * 1024


# Known Claude/Anthropic API error strings (as surfaced by the CLI in the
# ResultMessage ``result`` field) mapped to OpenAI-compatible error envelopes.
# Matching is a lowercased substring so minor CLI wording changes don't break it.
# Each entry is (needle, status_code, openai_error_type, openai_error_code).
_API_ERROR_SIGNATURES = [
    # The request exceeded the model's context window. This is the case behind
    # the raw ``Prompt is too long`` blob that previously leaked to clients.
    ("prompt is too long", 400, "invalid_request_error", "context_length_exceeded"),
    ("credit balance is too low", 400, "invalid_request_error", "insufficient_quota"),
    ("rate limit", 429, "rate_limit_error", "rate_limit_exceeded"),
    ("overloaded", 529, "overloaded_error", "overloaded"),
]


def classify_sdk_error(message: str, api_error_status: Optional[int] = None) -> Dict[str, Any]:
    """Map an SDK/CLI error string to an OpenAI-style error envelope.

    Returns a dict with ``message``, ``status_code``, ``type`` and ``code``.
    Recognizes a handful of well-known Anthropic API errors (notably
    "Prompt is too long", which means the request exceeded the model's context
    window) so callers can return an accurate HTTP status and error code instead
    of a generic 500 — or worse, a 200 with the error text as the reply. Falls
    back to ``api_error_status`` (when the CLI provided one) or 500.
    """
    text = (message or "").strip() or "Claude Code error"
    lowered = text.lower()
    for needle, status, err_type, code in _API_ERROR_SIGNATURES:
        if needle in lowered:
            return {"message": text, "status_code": status, "type": err_type, "code": code}

    if api_error_status:
        err_type = "api_error" if api_error_status >= 500 else "invalid_request_error"
        return {
            "message": text,
            "status_code": api_error_status,
            "type": err_type,
            "code": str(api_error_status),
        }

    return {"message": text, "status_code": 500, "type": "api_error", "code": "500"}


def find_sdk_error(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Scan SDK chunks for an error result and return a classified envelope.

    The CLI signals an upstream API failure with a ResultMessage carrying
    ``is_error=True`` (the human-readable reason is in ``result``; an HTTP status
    may be in ``api_error_status``). Note the CLI emits ``subtype="success"`` even
    for these API errors, so ``is_error`` — not the subtype — is the reliable
    signal. Our own subprocess-failure path uses ``error_during_execution`` with
    the reason in ``error_message``. Both are handled here so an error is never
    mistaken for a successful completion. Returns ``None`` when no chunk is an
    error.
    """
    for message in messages:
        is_error = bool(message.get("is_error"))
        subtype = message.get("subtype")
        if not (is_error or subtype in ("error_during_execution", "error_max_turns")):
            continue
        text = message.get("result") or message.get("error_message") or "Claude Code error"
        return classify_sdk_error(text, message.get("api_error_status"))
    return None


class ClaudeCodeCLI:
    def __init__(self, timeout: int = 600000, cwd: Optional[str] = None):
        self.timeout = timeout / 1000  # Convert ms to seconds
        self.temp_dir = None

        # If cwd is provided (from CLAUDE_CWD env var), use it
        # Otherwise create an isolated temp directory
        if cwd:
            self.cwd = Path(cwd)
            # Check if the directory exists
            if not self.cwd.exists():
                logger.error(f"ERROR: Specified working directory does not exist: {self.cwd}")
                logger.error(
                    "Please create the directory first or unset CLAUDE_CWD to use a temporary directory"
                )
                raise ValueError(f"Working directory does not exist: {self.cwd}")
            else:
                logger.info(f"Using CLAUDE_CWD: {self.cwd}")
        else:
            # Create isolated temp directory (cross-platform)
            self.temp_dir = tempfile.mkdtemp(prefix="claude_code_workspace_")
            self.cwd = Path(self.temp_dir)
            logger.info(f"Using temporary isolated workspace: {self.cwd}")

            # Register cleanup function to remove temp dir on exit
            atexit.register(self._cleanup_temp_dir)

        # Import auth manager
        from src.auth import auth_manager, validate_claude_code_auth

        # Validate authentication
        is_valid, auth_info = validate_claude_code_auth()
        if not is_valid:
            logger.warning(f"Claude Code authentication issues detected: {auth_info['errors']}")
        else:
            logger.info(f"Claude Code authentication method: {auth_info.get('method', 'unknown')}")

        # Store auth environment variables for SDK
        self.claude_env_vars = auth_manager.get_claude_code_env_vars()

    async def verify_cli(self) -> bool:
        """Verify Claude Agent SDK is working and authenticated."""
        try:
            # Test SDK with a simple query
            logger.info("Testing Claude Agent SDK...")

            messages = []
            async for message in query(
                prompt="Hello",
                options=ClaudeAgentOptions(
                    max_turns=1,
                    cwd=self.cwd,
                    system_prompt={"type": "preset", "preset": "claude_code"},
                ),
            ):
                messages.append(message)
                # Break early on first response to speed up verification
                # Handle both dict and object types
                msg_type = (
                    getattr(message, "type", None)
                    if hasattr(message, "type")
                    else message.get("type") if isinstance(message, dict) else None
                )
                if msg_type == "assistant":
                    break

            if messages:
                logger.info("✅ Claude Agent SDK verified successfully")
                return True
            else:
                logger.warning("⚠️ Claude Agent SDK test returned no messages")
                return False

        except Exception as e:
            logger.error(f"Claude Agent SDK verification failed: {e}")
            logger.warning("Please ensure Claude Code is installed and authenticated:")
            logger.warning("  1. Install: npm install -g @anthropic-ai/claude-code")
            logger.warning("  2. Set ANTHROPIC_API_KEY environment variable")
            logger.warning("  3. Test: claude --print 'Hello'")
            return False

    async def run_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        stream: bool = True,
        max_turns: int = 10,
        allowed_tools: Optional[List[str]] = None,
        disallowed_tools: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        continue_session: bool = False,
        permission_mode: Optional[str] = None,
        use_claude_code_preset: bool = True,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run Claude Agent using the Python SDK and yield response chunks."""

        try:
            # Set authentication environment variables (if any)
            original_env = {}
            if self.claude_env_vars:  # Only set env vars if we have any
                for key, value in self.claude_env_vars.items():
                    original_env[key] = os.environ.get(key)
                    os.environ[key] = value

            # Temp file holding an oversized system prompt (cleaned up below).
            system_prompt_file = None

            try:
                # Build SDK options
                options = ClaudeAgentOptions(max_turns=max_turns, cwd=self.cwd)

                # Set model if specified
                if model:
                    options.model = model

                # Set system prompt.
                # The Claude Agent SDK accepts either a plain string (passed to
                # the CLI as --system-prompt) or a preset dict. It does NOT
                # understand the Anthropic content-block form
                # {"type": "text", "text": ...}; that shape falls through the
                # SDK's handling and the system prompt is silently dropped, so a
                # custom prompt must be passed as a plain string.
                if system_prompt and len(system_prompt.encode("utf-8")) > MAX_CLI_ARG_BYTES:
                    # Too large for a single CLI argument: write it to a temp
                    # file and append it via --append-system-prompt-file (with an
                    # empty inline base). This fully provides the system prompt,
                    # exactly like the inline form, without the argv-length limit.
                    system_prompt_file = tempfile.NamedTemporaryFile(
                        mode="w",
                        suffix=".txt",
                        prefix="claude_system_prompt_",
                        delete=False,
                        encoding="utf-8",
                    )
                    system_prompt_file.write(system_prompt)
                    system_prompt_file.close()
                    options.system_prompt = ""
                    options.extra_args = {"append-system-prompt-file": system_prompt_file.name}
                    logger.info(
                        f"System prompt is large ({len(system_prompt)} chars); "
                        "passing via --append-system-prompt-file to avoid arg-length limits"
                    )
                elif system_prompt:
                    options.system_prompt = system_prompt
                elif use_claude_code_preset:
                    # Use Claude Code preset to maintain expected behavior
                    options.system_prompt = {"type": "preset", "preset": "claude_code"}
                else:
                    # Neutral system prompt: behave like a raw model call without
                    # injecting the (~18k token) Claude Code preset. Used by the
                    # Anthropic-compatible /v1/messages endpoint so callers that
                    # send no system prompt aren't silently bloated or steered.
                    options.system_prompt = ""

                # Set tool restrictions
                if allowed_tools:
                    options.allowed_tools = allowed_tools
                if disallowed_tools:
                    options.disallowed_tools = disallowed_tools

                # Set permission mode (needed for tool execution in API context)
                if permission_mode:
                    options.permission_mode = permission_mode

                # Handle session continuity
                if continue_session:
                    options.continue_conversation = True
                elif session_id:
                    options.resume = session_id

                # A very large prompt would also overflow the CLI argument limit
                # (it is passed positionally as "--print -- <prompt>"). When it
                # exceeds the threshold, feed it through stdin using the SDK's
                # streaming-input mode instead of the command line.
                if isinstance(prompt, str) and len(prompt.encode("utf-8")) > MAX_CLI_ARG_BYTES:

                    async def _prompt_stream(text=prompt):
                        yield {"type": "user", "message": {"role": "user", "content": text}}

                    query_prompt: Any = _prompt_stream()
                    logger.info(
                        f"Prompt is large ({len(prompt)} chars); "
                        "streaming via stdin to avoid arg-length limits"
                    )
                else:
                    query_prompt = prompt

                # Run the query and yield messages
                async for message in query(prompt=query_prompt, options=options):
                    # Debug logging
                    logger.debug(f"Raw SDK message type: {type(message)}")
                    logger.debug(f"Raw SDK message: {message}")

                    # Convert message object to dict if needed
                    if hasattr(message, "__dict__") and not isinstance(message, dict):
                        # Convert object to dict for consistent handling
                        message_dict = {}

                        # Get all attributes from the object
                        for attr_name in dir(message):
                            if not attr_name.startswith("_"):  # Skip private attributes
                                try:
                                    attr_value = getattr(message, attr_name)
                                    if not callable(attr_value):  # Skip methods
                                        message_dict[attr_name] = attr_value
                                except:
                                    pass

                        logger.debug(f"Converted message dict: {message_dict}")
                        yield message_dict
                    else:
                        yield message

            finally:
                # Remove the oversized-system-prompt temp file, if any.
                if system_prompt_file is not None:
                    try:
                        os.unlink(system_prompt_file.name)
                    except OSError:
                        pass

                # Restore original environment (if we changed anything)
                if original_env:
                    for key, original_value in original_env.items():
                        if original_value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = original_value

        except Exception as e:
            logger.error(f"Claude Agent SDK error: {e}")
            # Yield error message in the expected format
            yield {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "error_message": str(e),
            }

    def parse_claude_message(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Extract the assistant message from Claude Agent SDK messages.

        Prioritizes ResultMessage.result for multi-turn conversations,
        falls back to last AssistantMessage content.
        """
        # First, check for ResultMessage with 'result' field (multi-turn completion).
        # Skip error results: the CLI reports API failures (e.g. "Prompt is too
        # long") with subtype="success" but is_error=True, and the error text
        # lives in 'result'. Returning it here would surface an error as a normal
        # assistant reply. Callers detect errors separately via find_sdk_error().
        for message in messages:
            if (
                message.get("subtype") == "success"
                and "result" in message
                and not message.get("is_error")
            ):
                return message["result"]

        # Collect all text from AssistantMessages (take the last one with text)
        last_text = None
        for message in messages:
            # Look for AssistantMessage type (new SDK format)
            if "content" in message and isinstance(message["content"], list):
                text_parts = []
                for block in message["content"]:
                    # Handle TextBlock objects
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)

                if text_parts:
                    last_text = "\n".join(text_parts)

            # Fallback: look for old format
            elif message.get("type") == "assistant" and "message" in message:
                sdk_message = message["message"]
                if isinstance(sdk_message, dict) and "content" in sdk_message:
                    content = sdk_message["content"]
                    if isinstance(content, list) and len(content) > 0:
                        # Handle content blocks (Anthropic SDK format)
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                        if text_parts:
                            last_text = "\n".join(text_parts)
                    elif isinstance(content, str):
                        last_text = content

        return last_text

    @staticmethod
    def _tokens_from_usage(usage: Any) -> Optional[Dict[str, int]]:
        """Normalize the SDK/Anthropic ``usage`` payload into OpenAI-style token counts.

        ``usage`` may be a dict or an object. Prompt tokens include cache
        read/creation tokens so the count reflects total input consumption.
        Returns None if no usage data is present.
        """
        if usage is None:
            return None

        def _get(key: str) -> int:
            if isinstance(usage, dict):
                value = usage.get(key)
            else:
                value = getattr(usage, key, None)
            try:
                return int(value or 0)
            except (TypeError, ValueError):
                return 0

        input_tokens = _get("input_tokens")
        output_tokens = _get("output_tokens")
        cache_creation = _get("cache_creation_input_tokens")
        cache_read = _get("cache_read_input_tokens")

        if not any((input_tokens, output_tokens, cache_creation, cache_read)):
            return None

        prompt_tokens = input_tokens + cache_creation + cache_read
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": cache_creation,
            "cache_read_input_tokens": cache_read,
        }

    def extract_metadata(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract metadata like costs, tokens, and session info from SDK messages."""
        metadata = {
            "session_id": None,
            "total_cost_usd": 0.0,
            "duration_ms": 0,
            "num_turns": 0,
            "model": None,
            "usage": None,
        }

        for message in messages:
            # AssistantMessage carries the actual model that generated the response.
            # This is the most reliable source of the real model and overrides the
            # client-requested name.
            if message.get("model") and "content" in message:
                metadata["model"] = message["model"]

            # New SDK format - ResultMessage
            if message.get("subtype") == "success" and "total_cost_usd" in message:
                metadata.update(
                    {
                        "total_cost_usd": message.get("total_cost_usd", 0.0),
                        "duration_ms": message.get("duration_ms", 0),
                        "num_turns": message.get("num_turns", 0),
                        "session_id": message.get("session_id"),
                    }
                )
                tokens = self._tokens_from_usage(message.get("usage"))
                if tokens:
                    metadata["usage"] = tokens
            # New SDK format - SystemMessage
            elif message.get("subtype") == "init" and "data" in message:
                data = message["data"]
                metadata.update({"session_id": data.get("session_id"), "model": data.get("model")})
            # Old format fallback
            elif message.get("type") == "result":
                metadata.update(
                    {
                        "total_cost_usd": message.get("total_cost_usd", 0.0),
                        "duration_ms": message.get("duration_ms", 0),
                        "num_turns": message.get("num_turns", 0),
                        "session_id": message.get("session_id"),
                    }
                )
                tokens = self._tokens_from_usage(message.get("usage"))
                if tokens:
                    metadata["usage"] = tokens
            elif message.get("type") == "system" and message.get("subtype") == "init":
                metadata.update(
                    {"session_id": message.get("session_id"), "model": message.get("model")}
                )

        return metadata

    def estimate_token_usage(
        self, prompt: str, completion: str, model: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Estimate token usage based on character count.

        Uses rough approximation: ~4 characters per token for English text.
        This is approximate and may not match actual tokenization.
        """
        # Rough approximation: 1 token ≈ 4 characters
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(completion) // 4)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _cleanup_temp_dir(self):
        """Clean up temporary directory on exit."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary workspace: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory {self.temp_dir}: {e}")
