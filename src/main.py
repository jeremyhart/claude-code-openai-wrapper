import os
import json
import asyncio
import logging
import secrets
import string
import time
import uuid
from typing import Optional, AsyncGenerator, Dict, Any, List, Iterator, Literal
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
import httpx
from dotenv import load_dotenv

from src.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    Choice,
    Message,
    ToolCall,
    Usage,
    StreamChoice,
    SessionListResponse,
    ToolListResponse,
    ToolMetadataResponse,
    ToolConfigurationResponse,
    ToolConfigurationRequest,
    MCPServerConfigRequest,
    MCPServerInfoResponse,
    MCPServersListResponse,
    MCPConnectionRequest,
    # Anthropic API compatible models
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicTextBlock,
    AnthropicToolUseBlock,
    AnthropicUsage,
)
from src.claude_cli import ClaudeCodeCLI, find_sdk_error
from src.message_adapter import MessageAdapter
from src.function_calling import build_tool_prompt, parse_tool_calls, resolve_tools
from src.auth import (
    verify_api_key,
    security,
    validate_claude_code_auth,
    get_claude_code_auth_info,
    auth_manager,
)
from src.parameter_validator import ParameterValidator, CompatibilityReporter
from src.session_manager import session_manager
from src.tool_manager import tool_manager
from src.mcp_client import mcp_client, MCPServerConfig
from src.observability import setup_metrics, setup_loki_logging
from src.rate_limiter import (
    limiter,
    rate_limit_exceeded_handler,
    rate_limit_endpoint,
)
from datetime import datetime, timezone

from src import constants
from src.constants import (
    ANTHROPIC_MODELS_URL,
    ANTHROPIC_VERSION,
    CLAUDE_MODELS,
    CLAUDE_TOOLS,
    DEFAULT_ALLOWED_TOOLS,
    DEFAULT_MODEL_FALLBACK,
    MODEL_LIST_CACHE_TTL_SECONDS,
    MODEL_LIST_ERROR_TTL_SECONDS,
    MODEL_LIST_REQUEST_TIMEOUT_SECONDS,
)

# Load environment variables
load_dotenv()

# Configure logging based on debug mode
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() in ("true", "1", "yes", "on")
VERBOSE = os.getenv("VERBOSE", "false").lower() in ("true", "1", "yes", "on")

# Set logging level based on debug/verbose mode
log_level = logging.DEBUG if (DEBUG_MODE or VERBOSE) else logging.INFO
logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global variable to store runtime-generated API key
runtime_api_key = None

# Best-effort cache for Anthropic's live Models API.  The static constants remain
# the fallback so /v1/models keeps working for Claude CLI, Bedrock, Vertex, local
# development, and transient Anthropic API outages.
_model_list_cache: Dict[str, Any] = {"expires_at": 0.0, "models": None}
# Serializes cache refreshes so concurrent /v1/models requests at TTL expiry
# don't all stampede the upstream Anthropic API.
_model_list_lock = asyncio.Lock()


def _iso_to_unix(value: Any) -> Optional[int]:
    """Convert an Anthropic ISO-8601 'created_at' string to a unix timestamp."""
    if not isinstance(value, str):
        return None
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def _openai_model_from_anthropic(model_info: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an Anthropic ModelInfo object to OpenAI-compatible model metadata."""
    created = _iso_to_unix(model_info.get("created_at"))
    model: Dict[str, Any] = {
        "id": model_info["id"],
        "object": "model",
        "created": created if created is not None else int(datetime.now(timezone.utc).timestamp()),
        "owned_by": "anthropic",
    }

    # Preserve useful Anthropic metadata for clients that want it.  OpenAI clients
    # ignore unknown keys, and the existing id/object/owned_by shape is retained.
    for key in (
        "display_name",
        "created_at",
        "max_input_tokens",
        "max_tokens",
        "capabilities",
        "type",
    ):
        if key in model_info:
            model[key] = model_info[key]

    return model


def _fallback_model_payload() -> List[Dict[str, Any]]:
    now = int(datetime.now(timezone.utc).timestamp())
    return [
        {"id": model_id, "object": "model", "created": now, "owned_by": "anthropic"}
        for model_id in CLAUDE_MODELS
    ]


async def _fetch_anthropic_models() -> Optional[List[Dict[str, Any]]]:
    """Fetch all available models from Anthropic, returning None on fallback-worthy errors."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    headers = {
        "anthropic-version": ANTHROPIC_VERSION,
        "x-api-key": api_key,
    }
    beta_header = os.getenv("ANTHROPIC_BETA") or os.getenv("ANTHROPIC_BETA_HEADER")
    if beta_header:
        headers["anthropic-beta"] = beta_header

    params: Dict[str, Any] = {"limit": 1000}
    models: List[Dict[str, Any]] = []

    try:
        async with httpx.AsyncClient(timeout=MODEL_LIST_REQUEST_TIMEOUT_SECONDS) as client:
            while True:
                response = await client.get(ANTHROPIC_MODELS_URL, headers=headers, params=params)
                response.raise_for_status()
                payload = response.json()
                models.extend(
                    _openai_model_from_anthropic(model)
                    for model in payload.get("data", [])
                    if model.get("id")
                )

                if not payload.get("has_more") or not payload.get("last_id"):
                    break
                params["after_id"] = payload["last_id"]
    except Exception as exc:  # noqa: BLE001 - endpoint should degrade gracefully
        logger.warning("Failed to fetch Anthropic model list, using fallback: %s", exc)
        return None

    return models or None


async def get_available_models() -> List[Dict[str, Any]]:
    """Return live Anthropic models when possible, with cached static fallback."""
    if os.getenv("CLAUDE_MODELS_OVERRIDE", "").strip():
        return _fallback_model_payload()

    now = time.time()
    cached_models = _model_list_cache.get("models")
    if cached_models and now < float(_model_list_cache.get("expires_at", 0)):
        return cached_models

    async with _model_list_lock:
        # Recheck inside the lock so the first waiter populates the cache and
        # subsequent waiters return without re-fetching.
        now = time.time()
        cached_models = _model_list_cache.get("models")
        if cached_models and now < float(_model_list_cache.get("expires_at", 0)):
            return cached_models

        live_models = await _fetch_anthropic_models()
        if live_models:
            _model_list_cache.update(
                {"models": live_models, "expires_at": now + MODEL_LIST_CACHE_TTL_SECONDS}
            )
            return live_models

        fallback_models = _fallback_model_payload()
        # Use a short TTL on failure so transient outages don't suppress live
        # discovery for the full MODEL_LIST_CACHE_TTL_SECONDS window.
        _model_list_cache.update(
            {"models": fallback_models, "expires_at": now + MODEL_LIST_ERROR_TTL_SECONDS}
        )
        return fallback_models


def _pick_latest_sonnet(models: List[Dict[str, Any]]) -> Optional[str]:
    """Return the id of the newest Sonnet model in `models`, or None."""
    sonnets = [m for m in models if isinstance(m.get("id"), str) and "sonnet" in m["id"].lower()]
    if not sonnets:
        return None
    # Prefer Anthropic-provided created_at; fall back to the int `created` we set,
    # then to id-sort (date-suffixed ids sort correctly newest-last).
    sonnets.sort(
        key=lambda m: (
            _iso_to_unix(m.get("created_at")) or m.get("created") or 0,
            m["id"],
        )
    )
    return sonnets[-1]["id"]


async def resolve_default_model() -> Optional[str]:
    """Pick the latest Sonnet from /v1/models and store it as the default.

    Skipped when the operator pinned DEFAULT_MODEL via env var, or when no
    ANTHROPIC_API_KEY is configured (live discovery is the only auth-aware
    path; Bedrock, Vertex, and Claude CLI subscription users get the static
    DEFAULT_MODEL_FALLBACK).
    """
    if constants.DEFAULT_MODEL_ENV:
        return constants.DEFAULT_MODEL_ENV

    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.info(
            "Live model discovery disabled (no ANTHROPIC_API_KEY); " "using fallback default %s",
            DEFAULT_MODEL_FALLBACK,
        )
        return None

    try:
        models = await get_available_models()
    except Exception as exc:  # noqa: BLE001 - startup should never abort on this
        logger.warning("Could not resolve default model from /v1/models: %s", exc)
        return None

    latest = _pick_latest_sonnet(models)
    if latest:
        constants.RESOLVED_DEFAULT_MODEL = latest
        logger.info("Resolved default model from Anthropic Models API: %s", latest)
        return latest

    logger.info(
        "No Sonnet model found in /v1/models response; using fallback %s",
        DEFAULT_MODEL_FALLBACK,
    )
    return None


def generate_secure_token(length: int = 32) -> str:
    """Generate a secure random token for API authentication."""
    alphabet = string.ascii_letters + string.digits + "-_"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def prompt_for_api_protection() -> Optional[str]:
    """
    Interactively ask user if they want API key protection.
    Returns the generated token if user chooses protection, None otherwise.
    """
    # Don't prompt if API_KEY is already set via environment variable
    if os.getenv("API_KEY"):
        return None

    print("\n" + "=" * 60)
    print("🔐 API Endpoint Security Configuration")
    print("=" * 60)
    print("Would you like to protect your API endpoint with an API key?")
    print("This adds a security layer when accessing your server remotely.")
    print("")

    while True:
        try:
            choice = input("Enable API key protection? (y/N): ").strip().lower()

            if choice in ["", "n", "no"]:
                print("✅ API endpoint will be accessible without authentication")
                print("=" * 60)
                return None

            elif choice in ["y", "yes"]:
                token = generate_secure_token()
                print("")
                print("🔑 API Key Generated!")
                print("=" * 60)
                print(f"API Key: {token}")
                print("=" * 60)
                print("📋 IMPORTANT: Save this key - you'll need it for API calls!")
                print("   Example usage:")
                print(f'   curl -H "Authorization: Bearer {token}" \\')
                print("        http://localhost:8000/v1/models")
                print("=" * 60)
                return token

            else:
                print("Please enter 'y' for yes or 'n' for no (or press Enter for no)")

        except (EOFError, KeyboardInterrupt):
            print("\n✅ Defaulting to no authentication")
            return None


# Initialize Claude CLI
claude_cli = ClaudeCodeCLI(
    timeout=int(os.getenv("MAX_TIMEOUT", "600000")), cwd=os.getenv("CLAUDE_CWD")
)


class ClaudeProxyError(HTTPException):
    """HTTPException that carries an OpenAI-style error ``type`` and ``code``.

    Raised when the Claude SDK/CLI reports an upstream API failure (e.g. a
    "Prompt is too long" context-window error) so the response is a well-formed
    OpenAI error envelope with an accurate status, type and code instead of a
    generic 500 or — as before — a 200 whose content is the raw error string.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_type: str = "api_error",
        code: Optional[str] = None,
    ):
        super().__init__(status_code=status_code, detail=message)
        self.error_type = error_type
        self.code = code or str(status_code)


def _raise_for_sdk_error(chunks: List[Dict[str, Any]]) -> None:
    """Raise a :class:`ClaudeProxyError` if the SDK chunks contain an error result.

    No-op when the run succeeded. Centralizes the non-streaming error check so the
    chat-completions and messages endpoints fail the same way.
    """
    sdk_error = find_sdk_error(chunks)
    if sdk_error is None:
        return
    logger.error(
        f"Claude SDK error: {sdk_error['message']} "
        f"(status={sdk_error['status_code']}, code={sdk_error['code']})"
    )
    raise ClaudeProxyError(
        message=sdk_error["message"],
        status_code=sdk_error["status_code"],
        error_type=sdk_error["type"],
        code=sdk_error["code"],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Verify Claude Code authentication and CLI on startup."""
    # Wire up direct Loki log shipping if LOKI_URL is configured. Done first so the
    # rest of startup is captured. Handle is closed on shutdown to flush the buffer.
    loki_handler = setup_loki_logging()

    logger.info("Verifying Claude Code authentication and CLI...")

    # Validate authentication first
    auth_valid, auth_info = validate_claude_code_auth()

    if not auth_valid:
        logger.error("❌ Claude Code authentication failed!")
        for error in auth_info.get("errors", []):
            logger.error(f"  - {error}")
        logger.warning("Authentication setup guide:")
        logger.warning(
            "  1. For Claude subscription (Pro/Max): Run `claude setup-token` and set "
            "CLAUDE_CODE_OAUTH_TOKEN"
        )
        logger.warning("  2. For Anthropic API: Set ANTHROPIC_API_KEY")
        logger.warning("  3. For Bedrock: Set CLAUDE_CODE_USE_BEDROCK=1 + AWS credentials")
        logger.warning("  4. For Vertex AI: Set CLAUDE_CODE_USE_VERTEX=1 + GCP credentials")
    else:
        logger.info(f"✅ Claude Code authentication validated: {auth_info['method']}")

    # Verify Claude Agent SDK with timeout for graceful degradation
    try:
        logger.info("Testing Claude Agent SDK connection...")
        # Use asyncio.wait_for to enforce timeout (30 seconds)
        cli_verified = await asyncio.wait_for(claude_cli.verify_cli(), timeout=30.0)

        if cli_verified:
            logger.info("✅ Claude Agent SDK verified successfully")
        else:
            logger.warning("⚠️  Claude Agent SDK verification returned False")
            logger.warning("The server will start, but requests may fail.")
    except asyncio.TimeoutError:
        logger.warning("⚠️  Claude Agent SDK verification timed out (30s)")
        logger.warning("This may indicate network issues or SDK configuration problems.")
        logger.warning("The server will start, but first request may be slow.")
    except Exception as e:
        logger.error(f"⚠️  Claude Agent SDK verification failed: {e}")
        logger.warning("The server will start, but requests may fail.")
        logger.warning("Check that Claude Code CLI is properly installed and authenticated.")

    # Log debug information if debug mode is enabled
    if DEBUG_MODE or VERBOSE:
        logger.debug("🔧 Debug mode enabled - Enhanced logging active")
        logger.debug("🔧 Environment variables:")
        logger.debug(f"   DEBUG_MODE: {DEBUG_MODE}")
        logger.debug(f"   VERBOSE: {VERBOSE}")
        logger.debug(f"   PORT: {os.getenv('PORT', '8000')}")
        cors_origins_val = os.getenv("CORS_ORIGINS", '["*"]')
        logger.debug(f"   CORS_ORIGINS: {cors_origins_val}")
        logger.debug(f"   MAX_TIMEOUT: {os.getenv('MAX_TIMEOUT', '600000')}")
        logger.debug(f"   CLAUDE_CWD: {os.getenv('CLAUDE_CWD', 'Not set')}")
        logger.debug("🔧 Available endpoints:")
        logger.debug("   POST /v1/chat/completions - Main chat endpoint")
        logger.debug("   GET  /v1/models - List available models")
        logger.debug("   POST /v1/debug/request - Debug request validation")
        logger.debug("   GET  /v1/auth/status - Authentication status")
        logger.debug("   GET  /health - Health check")
        logger.debug(
            f"🔧 API Key protection: {'Enabled' if (os.getenv('API_KEY') or runtime_api_key) else 'Disabled'}"
        )

    # Resolve the default model from the live Anthropic Models API so /v1/chat
    # uses the latest Sonnet without a code change. Best-effort: any failure
    # leaves the static fallback in place.
    try:
        await resolve_default_model()
    except Exception as e:
        logger.warning(f"Default model resolution skipped: {e}")

    # Start session cleanup task
    session_manager.start_cleanup_task()

    yield

    # Cleanup on shutdown
    logger.info("Shutting down session manager...")
    session_manager.shutdown()

    # Flush and close the Loki handler so buffered logs are shipped before exit.
    if loki_handler is not None:
        logging.getLogger().removeHandler(loki_handler)
        loki_handler.close()


# Create FastAPI app
app = FastAPI(
    title="Claude Code OpenAI API Wrapper",
    description="OpenAI-compatible API for Claude Code",
    version="1.0.0",
    lifespan=lifespan,
)

# Expose Prometheus metrics at /metrics (opt-out via METRICS_ENABLED=false)
setup_metrics(app)

# Configure CORS
cors_origins = json.loads(os.getenv("CORS_ORIGINS", '["*"]'))
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting error handler
if limiter:
    app.state.limiter = limiter
    app.add_exception_handler(429, rate_limit_exceeded_handler)

# Security configuration
MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE", str(10 * 1024 * 1024)))  # 10MB default

# Add middleware
from starlette.middleware.base import BaseHTTPMiddleware


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request for audit trails."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Limit request body size to prevent DoS attacks."""

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_REQUEST_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "message": f"Request body too large. Maximum size is {MAX_REQUEST_SIZE} bytes.",
                        "type": "request_too_large",
                        "code": 413,
                    }
                },
            )
        return await call_next(request)


# Add security middleware (order matters - first added = last executed)
# RequestIDMiddleware is registered last (below) so it is the outermost layer and
# assigns request.state.request_id before the logging middleware reads it.
app.add_middleware(RequestSizeLimitMiddleware)


class DebugLoggingMiddleware(BaseHTTPMiddleware):
    """ASGI-compliant middleware for logging request/response details when debug mode is enabled."""

    async def dispatch(self, request: Request, call_next):
        # Get request ID for correlation
        request_id = getattr(request.state, "request_id", "unknown")
        start_time = asyncio.get_event_loop().time()

        # Verbose body/header logging only in debug mode.
        if DEBUG_MODE or VERBOSE:
            logger.debug(f"🔍 [{request_id}] Incoming request: {request.method} {request.url}")
            logger.debug(f"🔍 [{request_id}] Headers: {dict(request.headers)}")

            body_logged = False
            if request.method == "POST" and request.url.path.startswith("/v1/"):
                try:
                    content_length = request.headers.get("content-length")
                    if content_length and int(content_length) < 100000:  # Less than 100KB
                        body = await request.body()
                        if body:
                            try:
                                import json as json_lib

                                parsed_body = json_lib.loads(body.decode())
                                logger.debug(
                                    f"🔍 Request body: {json_lib.dumps(parsed_body, indent=2)}"
                                )
                                body_logged = True
                            except:
                                logger.debug(f"🔍 Request body (raw): {body.decode()[:500]}...")
                                body_logged = True
                except Exception as e:
                    logger.debug(f"🔍 Could not read request body: {e}")

            if not body_logged and request.method == "POST":
                logger.debug("🔍 Request body: [not logged - streaming or large payload]")

        # Only emit access logs for API traffic; skip health/metrics scrape noise.
        path = request.url.path
        is_api_request = path.startswith("/v1/")

        try:
            response = await call_next(request)

            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000

            # Always-on access log so a Grafana/Loki dashboard can confirm the proxy
            # is serving requests, with status and latency per call.
            if is_api_request:
                logger.info(
                    f"{request.method} {path} -> {response.status_code} in {duration_ms:.0f}ms",
                    extra={
                        "event": "http_access",
                        "request_id": request_id,
                        "method": request.method,
                        "path": path,
                        "status_code": response.status_code,
                        "duration_ms": round(duration_ms, 1),
                    },
                )
            else:
                logger.debug(f"🔍 Response: {response.status_code} in {duration_ms:.2f}ms")

            return response

        except Exception as e:
            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            logger.error(
                f"{request.method} {path} failed after {duration_ms:.0f}ms: {e}",
                extra={
                    "event": "http_error",
                    "request_id": request_id,
                    "method": request.method,
                    "path": path,
                    "duration_ms": round(duration_ms, 1),
                },
            )
            raise


# Add the debug/access logging middleware, then the request-ID middleware last so
# it wraps everything and request_id is available to the logging middleware above.
app.add_middleware(DebugLoggingMiddleware)
app.add_middleware(RequestIDMiddleware)


# Custom exception handler for 422 validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors with detailed debugging information."""

    # Log the validation error details
    logger.error(f"❌ Request validation failed for {request.method} {request.url}")
    logger.error(f"❌ Validation errors: {exc.errors()}")

    # Create detailed error response
    error_details = []
    for error in exc.errors():
        location = " -> ".join(str(loc) for loc in error.get("loc", []))
        error_details.append(
            {
                "field": location,
                "message": error.get("msg", "Unknown validation error"),
                "type": error.get("type", "validation_error"),
                "input": error.get("input"),
            }
        )

    # If debug mode is enabled, include the raw request body
    debug_info = {}
    if DEBUG_MODE or VERBOSE:
        try:
            body = await request.body()
            if body:
                debug_info["raw_request_body"] = body.decode()
        except:
            debug_info["raw_request_body"] = "Could not read request body"

    error_response = {
        "error": {
            "message": "Request validation failed - the request body doesn't match the expected format",
            "type": "validation_error",
            "code": "invalid_request_error",
            "details": error_details,
            "help": {
                "common_issues": [
                    "Missing required fields (model, messages)",
                    "Invalid field types (e.g. messages should be an array)",
                    "Invalid role values (must be 'system', 'user', or 'assistant')",
                    "Invalid parameter ranges (e.g. temperature must be 0-2)",
                ],
                "debug_tip": "Set DEBUG_MODE=true or VERBOSE=true environment variable for more detailed logging",
            },
        }
    }

    # Add debug info if available
    if debug_info:
        error_response["error"]["debug"] = debug_info

    return JSONResponse(status_code=422, content=error_response)


def _log_claude_proxy_start(
    request_id, requested_model, session_id, auth_method, endpoint, streaming
):
    """Emit the structured 'request forwarded to Claude' event used by the dashboard."""
    logger.info(
        f"Proxying {'streaming ' if streaming else ''}completion to Claude (model={requested_model})",
        extra={
            "event": "claude_proxy_start",
            "request_id": request_id,
            "model": requested_model,
            "session_id": session_id,
            "auth_method": auth_method,
            "endpoint": endpoint,
            "streaming": streaming,
        },
    )


def _log_claude_proxy_success(
    request_id,
    requested_model,
    metadata,
    prompt,
    completion_text,
    session_id,
    duration_ms,
    endpoint,
    streaming,
):
    """Emit the structured success event with real SDK usage/cost (falling back to estimate).

    Returns (prompt_tokens, completion_tokens, cost_usd) so callers can reuse them.
    """
    sdk_usage = metadata.get("usage")
    if sdk_usage:
        prompt_tokens = sdk_usage["prompt_tokens"]
        completion_tokens = sdk_usage["completion_tokens"]
        tokens_source = "sdk"
    else:
        prompt_tokens = MessageAdapter.estimate_tokens(prompt)
        completion_tokens = MessageAdapter.estimate_tokens(completion_text or "")
        tokens_source = "estimate"
    cost_usd = metadata.get("total_cost_usd") or 0.0
    # The model the SDK actually ran, not the (possibly bogus) client-requested name.
    resolved_model = metadata.get("model") or requested_model
    logger.info(
        f"Claude completion succeeded (model={resolved_model}, "
        f"{prompt_tokens + completion_tokens} tokens, ${cost_usd:.4f}, {duration_ms:.0f}ms)",
        extra={
            "event": "claude_proxy_success",
            "request_id": request_id,
            "model": resolved_model,
            "requested_model": requested_model,
            "session_id": session_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": round(cost_usd, 6),
            "tokens_source": tokens_source,
            "endpoint": endpoint,
            "streaming": streaming,
            "claude_duration_ms": round(duration_ms, 1),
        },
    )
    return prompt_tokens, completion_tokens, cost_usd


async def _run_single_completion(
    *,
    request_id: str,
    proxy_model: str,
    prompt: str,
    system_prompt: Optional[str],
    claude_options: Dict[str, Any],
    session_id: Optional[str],
    tool_prompt: Optional[str] = None,
):
    """Run one non-streaming completion.

    Returns ``(assistant_content, parsed_tool_calls, prompt_tokens,
    completion_tokens)``. ``parsed_tool_calls`` is ``None`` unless ``tool_prompt``
    was supplied AND the raw response parsed into a prompt-based tool call.

    This isolates the per-choice work so the n>1 path can call it repeatedly.
    It does NOT persist anything to the session; the caller decides which choice
    (the first one only) gets appended to the session history.
    """
    claude_start = asyncio.get_event_loop().time()
    chunks = []
    async for chunk in claude_cli.run_completion(
        prompt=prompt,
        system_prompt=system_prompt,
        model=claude_options.get("model"),
        max_turns=claude_options.get("max_turns", 10),
        allowed_tools=claude_options.get("allowed_tools"),
        disallowed_tools=claude_options.get("disallowed_tools"),
        permission_mode=claude_options.get("permission_mode"),
        stream=False,
    ):
        chunks.append(chunk)

    # Surface upstream API failures (e.g. "Prompt is too long") as proper errors
    # instead of letting the error text fall through as a successful reply.
    _raise_for_sdk_error(chunks)

    # Extract assistant message
    raw_assistant_content = claude_cli.parse_claude_message(chunks)

    if not raw_assistant_content:
        raise HTTPException(status_code=500, detail="No response from Claude Code")

    # If tools were requested, check the RAW content (before filtering, which
    # would strip the JSON envelope) for a prompt-based tool call.
    parsed_tool_calls = None
    if tool_prompt:
        parsed_tool_calls = parse_tool_calls(raw_assistant_content)

    # Filter out tool usage and thinking blocks
    assistant_content = MessageAdapter.filter_content(raw_assistant_content)

    # Log the structured success event with real SDK usage/cost (falling
    # back to estimate) and reuse the resolved token counts for the response.
    claude_duration_ms = (asyncio.get_event_loop().time() - claude_start) * 1000
    metadata = claude_cli.extract_metadata(chunks)
    prompt_tokens, completion_tokens, _cost = _log_claude_proxy_success(
        request_id=request_id,
        requested_model=proxy_model,
        metadata=metadata,
        prompt=prompt,
        completion_text=assistant_content,
        session_id=session_id,
        duration_ms=claude_duration_ms,
        endpoint="/v1/chat/completions",
        streaming=False,
    )
    return assistant_content, parsed_tool_calls, prompt_tokens, completion_tokens


# Maximum characters per streamed content delta. A single SDK text block can be
# large; splitting it into bounded segments lets clients render incrementally.
# 0 disables segmentation (emit each filtered block as one delta).
STREAM_MAX_DELTA_CHARS = int(os.getenv("STREAM_MAX_DELTA_CHARS", "0"))


async def generate_streaming_response(
    request: ChatCompletionRequest, request_id: str, claude_headers: Optional[Dict[str, Any]] = None
) -> AsyncGenerator[str, None]:
    """Generate SSE formatted streaming response."""
    try:
        # Process messages with session management
        all_messages, actual_session_id = session_manager.process_messages(
            request.messages, request.session_id
        )

        # Convert messages to prompt
        prompt, system_prompt = MessageAdapter.messages_to_prompt(all_messages)

        # Add sampling instructions from temperature/top_p if present
        sampling_instructions = request.get_sampling_instructions()
        if sampling_instructions:
            if system_prompt:
                system_prompt = f"{system_prompt}\n\n{sampling_instructions}"
            else:
                system_prompt = sampling_instructions
            logger.debug(f"Added sampling instructions: {sampling_instructions}")

        # Append the prompt-based function-calling fragment if tools/functions
        # were supplied. Composed AFTER sampling instructions (just appended).
        effective_tools, effective_tool_choice = resolve_tools(
            request.tools,
            request.tool_choice,
            request.functions,
            request.function_call,
        )
        tool_prompt = build_tool_prompt(effective_tools, effective_tool_choice)
        if tool_prompt:
            if system_prompt:
                system_prompt = f"{system_prompt}\n\n{tool_prompt}"
            else:
                system_prompt = tool_prompt
            logger.info("Added prompt-based function-calling instructions (streaming)")

        # Filter content for unsupported features
        prompt = MessageAdapter.filter_content(prompt)
        if system_prompt:
            system_prompt = MessageAdapter.filter_content(system_prompt)

        # Get Claude Agent SDK options from request
        claude_options = request.to_claude_options()

        # Merge with Claude-specific headers if provided
        if claude_headers:
            claude_options.update(claude_headers)

        # Validate model
        if claude_options.get("model"):
            ParameterValidator.validate_model(claude_options["model"])

        # Handle tools - disabled by default for OpenAI compatibility
        if not request.enable_tools:
            # Disable all tools by using CLAUDE_TOOLS constant
            claude_options["disallowed_tools"] = CLAUDE_TOOLS
            claude_options["max_turns"] = 1  # Single turn for Q&A
            logger.info("Tools disabled (default behavior for OpenAI compatibility)")
        else:
            # Enable tools - use default safe subset (Read, Glob, Grep, Bash, Write, Edit)
            claude_options["allowed_tools"] = DEFAULT_ALLOWED_TOOLS
            # Set permission mode to bypass prompts (required for API/headless usage)
            claude_options["permission_mode"] = "bypassPermissions"
            logger.info(f"Tools enabled by user request: {DEFAULT_ALLOWED_TOOLS}")

        # Run Claude Code
        proxy_model = claude_options.get("model") or request.model
        _log_claude_proxy_start(
            request_id,
            proxy_model,
            actual_session_id,
            auth_manager.auth_method,
            "/v1/chat/completions",
            streaming=True,
        )

        # n>1: stream the n choices sequentially. Each choice carries its own
        # choice.index across role/content/finish chunks (OpenAI-shaped). For
        # n==1 (and no tools) this produces output byte-for-byte identical to the
        # prior single stream. The aggregated usage/[DONE] handling lives after
        # the loop so a single trailing usage chunk is emitted for the response.
        n_choices = request.n or 1
        # For n==1 we keep the historical behavior of attaching usage directly to
        # the single finish chunk. For n>1 each choice emits its own finish chunk
        # (no usage) and a single aggregated usage chunk is emitted at the end.
        attach_usage_to_finish = n_choices == 1

        def _build_role_chunk(index: int) -> str:
            """SSE line for the initial role delta (sent exactly once per choice)."""
            initial_chunk = ChatCompletionStreamResponse(
                id=request_id,
                model=request.model,
                choices=[
                    StreamChoice(
                        index=index,
                        delta={"role": "assistant", "content": ""},
                        finish_reason=None,
                    )
                ],
            )
            return f"data: {initial_chunk.model_dump_json()}\n\n"

        def _build_content_chunk(index: int, text: str) -> str:
            """SSE line for a single non-empty content delta."""
            stream_chunk = ChatCompletionStreamResponse(
                id=request_id,
                model=request.model,
                choices=[StreamChoice(index=index, delta={"content": text}, finish_reason=None)],
            )
            return f"data: {stream_chunk.model_dump_json()}\n\n"

        def _iter_block_text(content: Any) -> Iterator[str]:
            """Yield raw text from an AssistantMessage content payload.

            Accepts the SDK's list-of-blocks form (TextBlock objects or dicts)
            as well as a bare string. Non-text blocks (tool use, etc.) are
            skipped so they never reach the wire.
            """
            blocks = content if isinstance(content, list) else [content]
            for block in blocks:
                if hasattr(block, "text"):
                    yield block.text
                elif isinstance(block, dict) and block.get("type") == "text":
                    yield block.get("text", "")
                elif isinstance(block, str):
                    yield block

        collected_completions: List[str] = []

        async def _stream_one_choice(index: int):
            """Stream a single choice, folding in enhanced-streaming + tools."""
            claude_start = asyncio.get_event_loop().time()
            chunks_buffer = []
            role_sent = False  # Track if we've sent the initial role chunk
            content_sent = False  # Track if we've sent any content
            stream_error_message = None  # Set if the SDK yields an error result

            async for chunk in claude_cli.run_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                model=claude_options.get("model"),
                max_turns=claude_options.get("max_turns", 10),
                allowed_tools=claude_options.get("allowed_tools"),
                disallowed_tools=claude_options.get("disallowed_tools"),
                permission_mode=claude_options.get("permission_mode"),
                stream=True,
            ):
                chunks_buffer.append(chunk)

                # Detect an error result yielded by the SDK so we can terminate
                # the stream cleanly rather than letting it surface as a throw.
                sdk_error = find_sdk_error([chunk])
                if sdk_error is not None:
                    stream_error_message = sdk_error["message"]
                    logger.error(f"SDK error during streaming: {stream_error_message}")
                    break

                # Only AssistantMessage chunks carry incremental text to stream.
                # The terminating ResultMessage (subtype == "success" / has
                # total_cost_usd) repeats the same text in its ``result`` field;
                # streaming it too would duplicate the whole response, so we
                # ignore it here and use it only for session storage / metadata.
                is_result_message = (
                    chunk.get("subtype") == "success"
                    or "total_cost_usd" in chunk
                    or chunk.get("type") == "result"
                )
                content = None
                if not is_result_message:
                    if chunk.get("type") == "assistant" and "message" in chunk:
                        # Old format: {"type": "assistant", "message": {...}}
                        message = chunk["message"]
                        if isinstance(message, dict) and "content" in message:
                            content = message["content"]
                    elif "content" in chunk and isinstance(chunk["content"], list):
                        # New format: {"content": [TextBlock(...)]} (AssistantMessage)
                        content = chunk["content"]

                if content is None:
                    continue

                # When tools are requested we cannot know whether the response is
                # a tool call until the full text is available, so suppress
                # incremental content streaming and decide after the loop. Without
                # tools, behavior is byte-for-byte identical to before.
                if tool_prompt:
                    continue

                # Send initial role chunk before any content.
                if not role_sent:
                    yield _build_role_chunk(index)
                    role_sent = True

                for raw_text in _iter_block_text(content):
                    # Filter per complete block (streaming-safe variant): scrub
                    # tool/thinking/image markup while preserving inter-token
                    # whitespace and never injecting placeholder fallback text.
                    filtered_text = MessageAdapter.filter_content_streaming(raw_text)
                    if not filtered_text or filtered_text.isspace():
                        # Skip empty / whitespace-only deltas; emitting them can
                        # confuse strict OpenAI clients.
                        continue
                    # Optionally smooth a very large block into multiple deltas.
                    for segment in MessageAdapter.segment_text(
                        filtered_text, STREAM_MAX_DELTA_CHARS
                    ):
                        if not segment:
                            continue
                        yield _build_content_chunk(index, segment)
                        content_sent = True

            # Extract assistant response from all chunks (used for tools, session,
            # metadata, and usage accounting regardless of the streaming path).
            assistant_content = None
            if chunks_buffer:
                assistant_content = claude_cli.parse_claude_message(chunks_buffer)

                # Store in session only for the FIRST choice so we don't append
                # multiple assistant turns for a single user request.
                if index == 0 and actual_session_id and assistant_content:
                    assistant_message = Message(role="assistant", content=assistant_content)
                    session_manager.add_assistant_response(actual_session_id, assistant_message)

            # Decide the per-choice finish reason. Default "stop"; switches to
            # "tool_calls" when tools were requested and a tool call was parsed.
            choice_finish_reason: Literal[
                "stop", "length", "content_filter", "tool_calls", "null"
            ] = "stop"

            if tool_prompt:
                # Decide whether the buffered response is a prompt-based tool
                # call. Parsing uses the RAW content (filter_content would strip
                # the JSON envelope).
                streamed_tool_calls = parse_tool_calls(assistant_content)
                if streamed_tool_calls:
                    choice_finish_reason = "tool_calls"
                    # Role chunk first, then a single delta carrying the whole
                    # tool_calls array (OpenAI-parseable).
                    role_chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        model=request.model,
                        choices=[
                            StreamChoice(
                                index=index,
                                delta={"role": "assistant", "content": None},
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {role_chunk.model_dump_json()}\n\n"

                    delta_tool_calls = [
                        {
                            "index": idx,
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"],
                            },
                        }
                        for idx, tc in enumerate(streamed_tool_calls)
                    ]
                    tool_chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        model=request.model,
                        choices=[
                            StreamChoice(
                                index=index,
                                delta={"tool_calls": delta_tool_calls},
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {tool_chunk.model_dump_json()}\n\n"
                else:
                    # No tool call detected: stream the buffered text as a normal
                    # response now (incremental emission was suppressed above).
                    yield _build_role_chunk(index)
                    role_sent = True
                    filtered = MessageAdapter.filter_content(assistant_content or "")
                    if filtered and not filtered.isspace():
                        yield _build_content_chunk(index, filtered)
                        content_sent = True
                    elif not content_sent:
                        # Keep the legacy non-empty guarantee for empty replies.
                        yield _build_content_chunk(
                            index, "I'm unable to provide a response at the moment."
                        )
                        content_sent = True
            else:
                # Always open the stream with a role delta, even if nothing
                # followed (no content, or an early error), so clients see a
                # well-formed stream.
                if not role_sent:
                    yield _build_role_chunk(index)
                    role_sent = True

                # If we sent the role but no content, send a single fallback
                # delta. For a mid-stream SDK error this surfaces the error text;
                # for an empty response it keeps the legacy placeholder behaviour.
                if role_sent and not content_sent:
                    fallback_text = (
                        f"An error occurred while generating the response: {stream_error_message}"
                        if stream_error_message
                        else "I'm unable to provide a response at the moment."
                    )
                    yield _build_content_chunk(index, fallback_text)

            # Emit the structured success event so streaming requests appear on
            # the dashboard with real token/cost/model data (first choice only
            # logging is not required; the n>1 branch logs each run).
            claude_duration_ms = (asyncio.get_event_loop().time() - claude_start) * 1000
            stream_metadata = claude_cli.extract_metadata(chunks_buffer)
            _log_claude_proxy_success(
                request_id=request_id,
                requested_model=proxy_model,
                metadata=stream_metadata,
                prompt=prompt,
                completion_text=assistant_content or "",
                session_id=actual_session_id,
                duration_ms=claude_duration_ms,
                endpoint="/v1/chat/completions",
                streaming=True,
            )

            # Stash the completion text for aggregated usage accounting.
            collected_completions.append(assistant_content or "")

            # Finish chunk. For n==1 (attach_usage_to_finish) usage rides on this
            # chunk exactly like before. For n>1 it's a bare per-choice finish.
            finish_usage = None
            if (
                attach_usage_to_finish
                and request.stream_options
                and request.stream_options.include_usage
            ):
                token_usage = claude_cli.estimate_token_usage(
                    prompt, assistant_content or "", request.model
                )
                finish_usage = Usage(
                    prompt_tokens=token_usage["prompt_tokens"],
                    completion_tokens=token_usage["completion_tokens"],
                    total_tokens=token_usage["total_tokens"],
                )
                logger.debug(f"Estimated usage: {finish_usage}")

            finish_chunk = ChatCompletionStreamResponse(
                id=request_id,
                model=request.model,
                choices=[StreamChoice(index=index, delta={}, finish_reason=choice_finish_reason)],
                usage=finish_usage,
            )
            yield f"data: {finish_chunk.model_dump_json()}\n\n"

        for index in range(n_choices):
            async for sse in _stream_one_choice(index):
                yield sse

        # For n>1, emit a single aggregated usage chunk after all choices when
        # usage was requested (OpenAI emits a trailing usage-only chunk).
        if (
            not attach_usage_to_finish
            and request.stream_options
            and request.stream_options.include_usage
        ):
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_tokens = 0
            for completion_text in collected_completions:
                token_usage = claude_cli.estimate_token_usage(
                    prompt, completion_text, request.model
                )
                total_prompt_tokens += token_usage["prompt_tokens"]
                total_completion_tokens += token_usage["completion_tokens"]
                total_tokens += token_usage["total_tokens"]
            usage_data = Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_tokens,
            )
            logger.debug(f"Estimated usage: {usage_data}")

            # Final usage-only chunk (empty choices), matching OpenAI's behavior
            # when stream_options.include_usage is set.
            usage_chunk = ChatCompletionStreamResponse(
                id=request_id,
                model=request.model,
                choices=[],
                usage=usage_data,
            )
            yield f"data: {usage_chunk.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_chunk = {"error": {"message": str(e), "type": "streaming_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"


@app.post("/v1/chat/completions")
@rate_limit_endpoint("chat")
async def chat_completions(
    request_body: ChatCompletionRequest,
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """OpenAI-compatible chat completions endpoint."""
    # Check FastAPI API key if configured
    await verify_api_key(request, credentials)

    # Validate Claude Code authentication
    auth_valid, auth_info = validate_claude_code_auth()

    if not auth_valid:
        error_detail = {
            "message": "Claude Code authentication failed",
            "errors": auth_info.get("errors", []),
            "method": auth_info.get("method", "none"),
            "help": "Check /v1/auth/status for detailed authentication information",
        }
        raise HTTPException(status_code=503, detail=error_detail)

    try:
        request_id = f"chatcmpl-{os.urandom(8).hex()}"

        # Extract Claude-specific parameters from headers
        claude_headers = ParameterValidator.extract_claude_headers(dict(request.headers))

        # Log compatibility info
        if logger.isEnabledFor(logging.DEBUG):
            compatibility_report = CompatibilityReporter.generate_compatibility_report(request_body)
            logger.debug(f"Compatibility report: {compatibility_report}")

        if request_body.stream:
            # Return streaming response
            return StreamingResponse(
                generate_streaming_response(request_body, request_id, claude_headers),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            # Non-streaming response
            # Process messages with session management
            all_messages, actual_session_id = session_manager.process_messages(
                request_body.messages, request_body.session_id
            )

            logger.info(
                f"Chat completion: session_id={actual_session_id}, total_messages={len(all_messages)}"
            )

            # Convert messages to prompt
            prompt, system_prompt = MessageAdapter.messages_to_prompt(all_messages)

            # Add sampling instructions from temperature/top_p if present
            sampling_instructions = request_body.get_sampling_instructions()
            if sampling_instructions:
                if system_prompt:
                    system_prompt = f"{system_prompt}\n\n{sampling_instructions}"
                else:
                    system_prompt = sampling_instructions
                logger.debug(f"Added sampling instructions: {sampling_instructions}")

            # Append the prompt-based function-calling fragment if tools/functions
            # were supplied. Composed AFTER sampling instructions (just appended).
            effective_tools, effective_tool_choice = resolve_tools(
                request_body.tools,
                request_body.tool_choice,
                request_body.functions,
                request_body.function_call,
            )
            tool_prompt = build_tool_prompt(effective_tools, effective_tool_choice)
            if tool_prompt:
                if system_prompt:
                    system_prompt = f"{system_prompt}\n\n{tool_prompt}"
                else:
                    system_prompt = tool_prompt
                logger.info("Added prompt-based function-calling instructions")

            # Filter content
            prompt = MessageAdapter.filter_content(prompt)
            if system_prompt:
                system_prompt = MessageAdapter.filter_content(system_prompt)

            # Get Claude Agent SDK options from request
            claude_options = request_body.to_claude_options()

            # Merge with Claude-specific headers
            if claude_headers:
                claude_options.update(claude_headers)

            # Validate model
            if claude_options.get("model"):
                ParameterValidator.validate_model(claude_options["model"])

            # Handle tools - disabled by default for OpenAI compatibility
            if not request_body.enable_tools:
                # Disable all tools by using CLAUDE_TOOLS constant
                claude_options["disallowed_tools"] = CLAUDE_TOOLS
                claude_options["max_turns"] = 1  # Single turn for Q&A
                logger.info("Tools disabled (default behavior for OpenAI compatibility)")
            else:
                # Enable tools - use default safe subset (Read, Glob, Grep, Bash, Write, Edit)
                claude_options["allowed_tools"] = DEFAULT_ALLOWED_TOOLS
                # Set permission mode to bypass prompts (required for API/headless usage)
                claude_options["permission_mode"] = "bypassPermissions"
                logger.info(f"Tools enabled by user request: {DEFAULT_ALLOWED_TOOLS}")

            # Collect all chunks
            proxy_model = claude_options.get("model") or request_body.model
            _log_claude_proxy_start(
                request_id,
                proxy_model,
                actual_session_id,
                auth_info.get("method"),
                "/v1/chat/completions",
                streaming=False,
            )

            # For n>1 we run the completion n times (sequentially). Each run
            # produces one choice; usage is summed across runs. Only the FIRST
            # choice is persisted to the session so we don't append multiple
            # assistant turns for a single user request. Function calling is
            # folded into each run via _run_single_completion (tool_prompt).
            n_choices = request_body.n or 1
            choices: List[Choice] = []
            total_prompt_tokens = 0
            total_completion_tokens = 0

            for index in range(n_choices):
                (
                    assistant_content,
                    parsed_tool_calls,
                    prompt_tokens,
                    completion_tokens,
                ) = await _run_single_completion(
                    request_id=request_id,
                    proxy_model=proxy_model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    claude_options=claude_options,
                    session_id=actual_session_id,
                    tool_prompt=tool_prompt,
                )

                # Add only the first assistant response to the session. For a
                # tool call there is no text content; store an empty string so
                # the turn is still recorded.
                if index == 0 and actual_session_id:
                    assistant_message = Message(
                        role="assistant", content=assistant_content if not parsed_tool_calls else ""
                    )
                    session_manager.add_assistant_response(actual_session_id, assistant_message)

                # If a prompt-based tool call was parsed, return it as OpenAI
                # tool_calls with content=null and finish_reason=tool_calls;
                # otherwise return the normal text response unchanged.
                if parsed_tool_calls:
                    choice_message = Message(
                        role="assistant",
                        content=None,
                        tool_calls=[ToolCall(**tc) for tc in parsed_tool_calls],
                    )
                    choices.append(
                        Choice(index=index, message=choice_message, finish_reason="tool_calls")
                    )
                else:
                    choices.append(
                        Choice(
                            index=index,
                            message=Message(role="assistant", content=assistant_content),
                            finish_reason="stop",
                        )
                    )
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens

            # Create response
            response = ChatCompletionResponse(
                id=request_id,
                model=request_body.model,
                choices=choices,
                usage=Usage(
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    total_tokens=total_prompt_tokens + total_completion_tokens,
                ),
            )

            return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _prepare_anthropic_request(
    request_body: AnthropicMessagesRequest,
) -> tuple[str, Optional[str], Optional[str]]:
    """Build the ``(prompt, system_prompt, tool_prompt)`` for /v1/messages.

    Flattens the Anthropic messages into a prompt, resolves the caller's
    system prompt, and folds the caller's tools into the system prompt via the
    shared prompt-based function-calling machinery. Shared by the streaming and
    non-streaming paths so they stay in lock-step.
    """
    messages = request_body.to_openai_messages()

    prompt_parts = []
    for msg in messages:
        if msg.role == "user":
            prompt_parts.append(msg.content)
        elif msg.role == "assistant":
            prompt_parts.append(f"Assistant: {msg.content}")
    prompt = "\n\n".join(prompt_parts)

    system_prompt = request_body.get_system_prompt()

    effective_tools, effective_tool_choice = resolve_tools(
        request_body.to_openai_tools(),
        request_body.to_openai_tool_choice(),
    )
    tool_prompt = build_tool_prompt(effective_tools, effective_tool_choice)
    if tool_prompt:
        system_prompt = f"{system_prompt}\n\n{tool_prompt}" if system_prompt else tool_prompt

    prompt = MessageAdapter.filter_content(prompt)
    if system_prompt:
        system_prompt = MessageAdapter.filter_content(system_prompt)

    return prompt, system_prompt, tool_prompt


def _anthropic_sse(event: str, data: Dict[str, Any]) -> str:
    """Format a single named Server-Sent Event in the Anthropic stream shape."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def generate_anthropic_streaming_response(
    request_body: AnthropicMessagesRequest, request_id: str, auth_method: Optional[str]
) -> AsyncGenerator[str, None]:
    """Generate Anthropic-format SSE for the /v1/messages endpoint.

    Emits the native Anthropic event sequence (``message_start`` →
    ``content_block_*`` → ``message_delta`` → ``message_stop``). Text responses
    stream incrementally; because tool calls are emulated via prompting, the
    full response must be buffered before we can tell a tool call from prose, so
    when tools are requested the content is emitted after the SDK stream drains.
    """
    model = request_body.model
    message_id = f"msg_{os.urandom(12).hex()}"
    try:
        prompt, system_prompt, tool_prompt = _prepare_anthropic_request(request_body)

        _log_claude_proxy_start(
            request_id, model, None, auth_method, "/v1/messages", streaming=True
        )
        claude_start = asyncio.get_event_loop().time()

        # Anthropic reports input tokens in message_start; estimate up front
        # (real token counts aren't known until the stream completes).
        input_tokens = claude_cli.estimate_token_usage(prompt, "", model)["prompt_tokens"]

        yield _anthropic_sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                },
            },
        )

        chunks_buffer: List[Dict[str, Any]] = []
        text_block_open = False
        stream_error_message = None

        async for chunk in claude_cli.run_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            max_turns=1,
            disallowed_tools=CLAUDE_TOOLS,
            use_claude_code_preset=False,
            stream=True,
        ):
            chunks_buffer.append(chunk)

            sdk_error = find_sdk_error([chunk])
            if sdk_error is not None:
                stream_error_message = sdk_error["message"]
                logger.error(f"SDK error during streaming: {stream_error_message}")
                break

            # Defer all emission until the buffer is complete when tools are in
            # play (can't distinguish a tool-call envelope mid-stream).
            if tool_prompt:
                continue

            is_result_message = (
                chunk.get("subtype") == "success"
                or "total_cost_usd" in chunk
                or chunk.get("type") == "result"
            )
            content = None
            if not is_result_message:
                if chunk.get("type") == "assistant" and "message" in chunk:
                    message = chunk["message"]
                    if isinstance(message, dict) and "content" in message:
                        content = message["content"]
                elif "content" in chunk and isinstance(chunk["content"], list):
                    content = chunk["content"]
            if content is None:
                continue

            blocks = content if isinstance(content, list) else [content]
            for block in blocks:
                if hasattr(block, "text"):
                    raw_text = block.text
                elif isinstance(block, dict) and block.get("type") == "text":
                    raw_text = block.get("text", "")
                elif isinstance(block, str):
                    raw_text = block
                else:
                    continue

                filtered_text = MessageAdapter.filter_content_streaming(raw_text)
                if not filtered_text or filtered_text.isspace():
                    continue
                if not text_block_open:
                    yield _anthropic_sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                    text_block_open = True
                for segment in MessageAdapter.segment_text(filtered_text, STREAM_MAX_DELTA_CHARS):
                    if not segment:
                        continue
                    yield _anthropic_sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": segment},
                        },
                    )

        # Buffered full response: used for tool detection, usage, and metadata.
        assistant_content = (
            claude_cli.parse_claude_message(chunks_buffer) if chunks_buffer else None
        )
        stop_reason = "end_turn"

        if tool_prompt:
            parsed_tool_calls = parse_tool_calls(assistant_content)
            if parsed_tool_calls:
                stop_reason = "tool_use"
                for idx, tc in enumerate(parsed_tool_calls):
                    fn = tc.get("function", {})
                    try:
                        tool_input = json.loads(fn.get("arguments", "{}"))
                    except (ValueError, TypeError):
                        tool_input = {}
                    yield _anthropic_sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": f"toolu_{os.urandom(12).hex()}",
                                "name": fn.get("name", ""),
                                "input": {},
                            },
                        },
                    )
                    yield _anthropic_sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": json.dumps(tool_input),
                            },
                        },
                    )
                    yield _anthropic_sse(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": idx},
                    )
            else:
                # Not a tool call: emit the buffered text as one text block.
                filtered = MessageAdapter.filter_content(assistant_content or "")
                yield _anthropic_sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                if filtered and not filtered.isspace():
                    yield _anthropic_sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": filtered},
                        },
                    )
                yield _anthropic_sse(
                    "content_block_stop", {"type": "content_block_stop", "index": 0}
                )
        else:
            # Text path: ensure a well-formed (possibly empty) block is closed.
            if not text_block_open:
                yield _anthropic_sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                text_block_open = True
                fallback_text = (
                    f"An error occurred while generating the response: {stream_error_message}"
                    if stream_error_message
                    else "I'm unable to provide a response at the moment."
                )
                yield _anthropic_sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": fallback_text},
                    },
                )
            yield _anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": 0})

        output_tokens = claude_cli.estimate_token_usage(prompt, assistant_content or "", model)[
            "completion_tokens"
        ]
        yield _anthropic_sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            },
        )
        yield _anthropic_sse("message_stop", {"type": "message_stop"})

        # Observability: surface the streamed request with real token/cost data.
        claude_duration_ms = (asyncio.get_event_loop().time() - claude_start) * 1000
        _log_claude_proxy_success(
            request_id=request_id,
            requested_model=model,
            metadata=claude_cli.extract_metadata(chunks_buffer),
            prompt=prompt,
            completion_text=assistant_content or "",
            session_id=None,
            duration_ms=claude_duration_ms,
            endpoint="/v1/messages",
            streaming=True,
        )

    except Exception as e:
        logger.error(f"Anthropic streaming error: {e}")
        yield _anthropic_sse(
            "error",
            {"type": "error", "error": {"type": "api_error", "message": str(e)}},
        )


@app.post("/v1/messages")
@rate_limit_endpoint("chat")
async def anthropic_messages(
    request_body: AnthropicMessagesRequest,
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Anthropic Messages API compatible endpoint.

    This endpoint provides compatibility with the native Anthropic SDK,
    allowing tools like VC to use this wrapper via the VC_API_BASE setting.
    """
    # Check FastAPI API key if configured
    await verify_api_key(request, credentials)

    # Validate Claude Code authentication
    auth_valid, auth_info = validate_claude_code_auth()

    if not auth_valid:
        error_detail = {
            "message": "Claude Code authentication failed",
            "errors": auth_info.get("errors", []),
            "method": auth_info.get("method", "none"),
            "help": "Check /v1/auth/status for detailed authentication information",
        }
        raise HTTPException(status_code=503, detail=error_detail)

    try:
        request_id = f"msg-{os.urandom(8).hex()}"
        logger.info(f"Anthropic Messages API request: model={request_body.model}")

        # Streaming: return the native Anthropic SSE event stream.
        if request_body.stream:
            return StreamingResponse(
                generate_anthropic_streaming_response(
                    request_body, request_id, auth_info.get("method")
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        # Honor the caller's system/tools via the shared preparation helper: the
        # caller's tools are emulated (not executed by the SDK) and folded into
        # the system prompt; the model's own (Claude Code) tools stay disabled.
        prompt, system_prompt, tool_prompt = _prepare_anthropic_request(request_body)

        # Run the model as a plain completion. This endpoint mirrors the native
        # Anthropic Messages API, so the caller's system/tools are authoritative:
        # the built-in Claude Code tools are disabled and the ~18k Claude Code
        # preset prompt is bypassed (use_claude_code_preset=False) so requests are
        # not silently bloated or steered by a hidden persona.
        _log_claude_proxy_start(
            request_id,
            request_body.model,
            None,
            auth_info.get("method"),
            "/v1/messages",
            streaming=False,
        )
        claude_start = asyncio.get_event_loop().time()
        chunks = []
        async for chunk in claude_cli.run_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            model=request_body.model,
            max_turns=1,
            disallowed_tools=CLAUDE_TOOLS,
            use_claude_code_preset=False,
            stream=False,
        ):
            chunks.append(chunk)

        # Surface upstream API failures (e.g. "Prompt is too long") as proper
        # errors instead of returning the error text as the assistant message.
        _raise_for_sdk_error(chunks)

        # Extract assistant message
        raw_assistant_content = claude_cli.parse_claude_message(chunks)

        if not raw_assistant_content:
            raise HTTPException(status_code=500, detail="No response from Claude Code")

        # Check the RAW content (before filtering, which strips the JSON
        # envelope) for a prompt-based tool call.
        parsed_tool_calls = parse_tool_calls(raw_assistant_content) if tool_prompt else None

        # Filter out tool usage and thinking blocks
        assistant_content = MessageAdapter.filter_content(raw_assistant_content)

        # Log the structured success event with real SDK usage/cost (or estimate).
        claude_duration_ms = (asyncio.get_event_loop().time() - claude_start) * 1000
        metadata = claude_cli.extract_metadata(chunks)
        prompt_tokens, completion_tokens, _cost = _log_claude_proxy_success(
            request_id=request_id,
            requested_model=request_body.model,
            metadata=metadata,
            prompt=prompt,
            completion_text=assistant_content,
            session_id=None,
            duration_ms=claude_duration_ms,
            endpoint="/v1/messages",
            streaming=False,
        )

        # Build the Anthropic-format response. A parsed tool call becomes one or
        # more tool_use blocks with stop_reason="tool_use"; otherwise it's a
        # normal text block with stop_reason="end_turn".
        if parsed_tool_calls:
            content_blocks = []
            for tc in parsed_tool_calls:
                fn = tc.get("function", {})
                try:
                    tool_input = json.loads(fn.get("arguments", "{}"))
                except (ValueError, TypeError):
                    tool_input = {}
                content_blocks.append(
                    AnthropicToolUseBlock(name=fn.get("name", ""), input=tool_input)
                )
            response = AnthropicMessagesResponse(
                model=request_body.model,
                content=content_blocks,
                stop_reason="tool_use",
                usage=AnthropicUsage(
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                ),
            )
        else:
            response = AnthropicMessagesResponse(
                model=request_body.model,
                content=[AnthropicTextBlock(text=assistant_content)],
                stop_reason="end_turn",
                usage=AnthropicUsage(
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                ),
            )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Anthropic Messages API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models(
    request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """List available models, preferring Anthropic's live Models API when configured."""
    # Check FastAPI API key if configured
    await verify_api_key(request, credentials)

    return {"object": "list", "data": await get_available_models()}


@app.post("/v1/compatibility")
async def check_compatibility(request_body: ChatCompletionRequest):
    """Check OpenAI API compatibility for a request."""
    report = CompatibilityReporter.generate_compatibility_report(request_body)
    return {
        "compatibility_report": report,
        "claude_agent_sdk_options": {
            "supported": [
                "model",
                "system_prompt",
                "max_turns",
                "allowed_tools",
                "disallowed_tools",
                "permission_mode",
                "max_thinking_tokens",
                "continue_conversation",
                "resume",
                "cwd",
            ],
            "custom_headers": [
                "X-Claude-Max-Turns",
                "X-Claude-Allowed-Tools",
                "X-Claude-Disallowed-Tools",
                "X-Claude-Permission-Mode",
                "X-Claude-Max-Thinking-Tokens",
            ],
        },
    }


@app.get("/health")
@rate_limit_endpoint("health")
async def health_check(request: Request):
    """Health check endpoint."""
    return {"status": "healthy", "service": "claude-code-openai-wrapper"}


@app.get("/version")
@rate_limit_endpoint("health")
async def version_info(request: Request):
    """Version information endpoint."""
    from src import __version__

    return {
        "version": __version__,
        "service": "claude-code-openai-wrapper",
        "api_version": "v1",
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    """Landing page with API documentation."""
    from src import __version__

    auth_info = get_claude_code_auth_info()
    auth_method = auth_info.get("method", "unknown")
    auth_valid = auth_info.get("status", {}).get("valid", False)
    status_color = "#22c55e" if auth_valid else "#ef4444"
    status_text = "Connected" if auth_valid else "Not Connected"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en" data-theme="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta name="color-scheme" content="light dark">
        <title>Claude Code OpenAI Wrapper</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">
        <style>
            :root {{
                --pico-font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                --accent-color: #16a34a;
            }}
            /* Light mode colors */
            [data-theme="light"] {{
                --card-bg: #ffffff;
                --subtle-bg: #f1f5f9;
                --border-color: #e2e8f0;
                --page-bg: #f8fafc;
            }}
            /* Dark mode colors */
            [data-theme="dark"] {{
                --card-bg: #1e293b;
                --subtle-bg: #334155;
                --border-color: #475569;
                --page-bg: #0f172a;
            }}
            /* Page background */
            body {{ background: var(--page-bg); }}
            /* GLOBAL FIX: Remove Pico's default code styling everywhere */
            code:not(pre code) {{
                background: transparent !important;
                padding: 0 !important;
                border-radius: 0 !important;
                color: inherit !important;
            }}
            /* Only style code green where we explicitly want it */
            .green-code {{ color: var(--accent-color) !important; }}
            /* Constrain page width - wider for modern screens */
            .container {{
                max-width: 1100px;
                margin: 0 auto;
                padding: 1.5rem 2rem;
            }}
            /* Override Pico article styling */
            article {{
                background: var(--card-bg);
                border: 1px solid var(--border-color);
                border-radius: 0.75rem;
                margin-bottom: 1rem;
                padding: 1rem 1.25rem;
            }}
            article header {{
                padding: 0;
                margin-bottom: 0.75rem;
                background: transparent;
                border: none;
            }}
            /* Section headers with icons - matches status-flex layout */
            .section-header {{
                display: flex;
                align-items: center;
                gap: 0.5rem;
                margin-bottom: 0.75rem;
            }}
            .section-icon {{
                width: 1rem;
                height: 1rem;
                color: var(--accent-color);
                flex-shrink: 0;
            }}
            /* Status indicator */
            .status-dot {{
                width: 0.75rem;
                height: 0.75rem;
                border-radius: 50%;
                display: inline-block;
                animation: pulse 2s infinite;
            }}
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.5; }}
            }}
            /* Method badges */
            .badge {{
                display: inline-block;
                padding: 0.25rem 0.5rem;
                font-size: 0.7rem;
                font-weight: 700;
                border-radius: 0.25rem;
                text-transform: uppercase;
            }}
            .badge-post {{ background: rgba(34, 197, 94, 0.15); color: #16a34a; }}
            .badge-get {{ background: rgba(59, 130, 246, 0.15); color: #2563eb; }}
            /* Header layout */
            .header-flex {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 1rem;
                margin-bottom: 1rem;
            }}
            .header-left {{
                display: flex;
                align-items: center;
                gap: 1rem;
                flex-shrink: 0;
            }}
            .header-right {{
                display: flex;
                align-items: center;
                gap: 0.75rem;
                flex-shrink: 0;
            }}
            .icon-btn {{
                padding: 0.5rem;
                border-radius: 0.5rem;
                background: var(--subtle-bg);
                border: 1px solid var(--border-color);
                cursor: pointer;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                color: inherit;
            }}
            .icon-btn:hover {{ opacity: 0.8; }}
            .icon-btn svg {{ width: 1.25rem; height: 1.25rem; }}
            .version-badge {{
                padding: 0.25rem 0.75rem;
                background: var(--subtle-bg);
                border: 1px solid var(--border-color);
                border-radius: 0.5rem;
                font-family: monospace;
                font-size: 0.875rem;
            }}
            /* Logo container */
            .logo-container {{
                background: linear-gradient(135deg, #22c55e 0%, #0ea5e9 100%);
                padding: 2px;
                border-radius: 0.75rem;
            }}
            .logo-inner {{
                background: var(--card-bg);
                border-radius: calc(0.75rem - 2px);
                padding: 0.75rem;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .logo-inner svg {{ width: 2rem; height: 2rem; color: #22c55e; }}
            /* Endpoint list */
            .endpoint-item {{
                display: flex;
                align-items: center;
                gap: 0.75rem;
                padding: 0.5rem 0;
                border-bottom: 1px solid var(--pico-muted-border-color);
            }}
            .endpoint-item:last-child {{ border-bottom: none; }}
            .endpoint-item code {{ flex: 1; }}
            .endpoint-desc {{ color: var(--pico-muted-color); font-size: 0.85rem; }}
            /* Details accordion styling */
            details {{
                border: 1px solid var(--border-color);
                border-radius: 0.5rem;
                margin-bottom: 0.4rem;
                background: var(--subtle-bg);
            }}
            details summary {{
                padding: 0.5rem 0.75rem;
                display: flex;
                align-items: center;
                gap: 0.75rem;
                cursor: pointer;
                list-style: none;
            }}
            details summary::-webkit-details-marker {{ display: none; }}
            details summary::after {{
                content: "";
                margin-left: auto;
                width: 0.5rem;
                height: 0.5rem;
                border-right: 2px solid currentColor;
                border-bottom: 2px solid currentColor;
                transform: rotate(-45deg);
                transition: transform 0.2s;
            }}
            details[open] summary::after {{ transform: rotate(45deg); }}
            details .content {{ padding: 0 1rem 1rem; }}
            details .content pre {{
                margin: 0;
                font-size: 0.875rem;
                overflow-x: auto;
            }}
            /* Config grid */
            .config-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                gap: 0.75rem;
            }}
            .config-item {{
                padding: 0.75rem;
                background: var(--subtle-bg);
                border: 1px solid var(--border-color);
                border-radius: 0.5rem;
            }}
            .config-item code {{ font-weight: 600; }}
            .config-item p {{ margin: 0.25rem 0 0; font-size: 0.875rem; color: var(--pico-muted-color); }}
            /* Footer */
            footer nav {{
                display: flex;
                justify-content: center;
                gap: 2rem;
            }}
            footer a {{
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }}
            footer svg {{ width: 1rem; height: 1rem; }}
            /* Quick start */
            .quickstart-wrapper {{ position: relative; }}
            .copy-btn {{
                position: absolute;
                top: 0.5rem;
                right: 0.5rem;
                padding: 0.5rem;
                background: var(--subtle-bg);
                border: 1px solid var(--border-color);
                border-radius: 0.5rem;
                cursor: pointer;
                z-index: 1;
                color: inherit;
            }}
            .copy-btn:hover {{ opacity: 0.8; }}
            .copy-btn svg {{ width: 1rem; height: 1rem; }}
            .hidden {{ display: none !important; }}
            /* Shiki code styling */
            .shiki {{ padding: 1rem; border-radius: 0.5rem; overflow-x: auto; }}
            .shiki code {{ white-space: pre-wrap; word-break: break-word; }}
            /* Status card layout */
            .status-flex {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 1rem;
            }}
            .status-left {{
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }}
            .auth-badge {{
                padding: 0.25rem 0.75rem;
                background: var(--subtle-bg);
                border: 1px solid var(--border-color);
                border-radius: 1rem;
                font-size: 0.875rem;
            }}
        </style>
        <script type="module">
            import {{ codeToHtml }} from 'https://esm.sh/shiki@3.0.0';

            const lightTheme = 'github-light';
            const darkTheme = 'github-dark';

            function isDark() {{
                return document.documentElement.getAttribute('data-theme') === 'dark';
            }}

            async function highlightJson(json, targetId) {{
                const code = typeof json === 'string' ? json : JSON.stringify(json, null, 2);
                const theme = isDark() ? darkTheme : lightTheme;
                try {{
                    const html = await codeToHtml(code, {{ lang: 'json', theme }});
                    document.getElementById(targetId).innerHTML = html;
                }} catch (e) {{
                    document.getElementById(targetId).innerHTML = '<pre style="color:red;">Error: ' + e.message + '</pre>';
                }}
            }}

            // Lazy load data when details opens
            document.querySelectorAll('details[data-endpoint]').forEach(details => {{
                details.addEventListener('toggle', async () => {{
                    if (details.open) {{
                        const id = details.id;
                        const endpoint = details.dataset.endpoint;
                        const dataContainer = document.getElementById('data-' + id);
                        const loader = document.getElementById('loader-' + id);
                        if (dataContainer.innerHTML === '' || dataContainer.dataset.theme !== (isDark() ? 'dark' : 'light')) {{
                            loader.classList.remove('hidden');
                            try {{
                                const response = await fetch(endpoint);
                                const json = await response.json();
                                await highlightJson(json, 'data-' + id);
                                dataContainer.dataset.theme = isDark() ? 'dark' : 'light';
                            }} catch (e) {{
                                dataContainer.innerHTML = '<span style="color:red;">Error: ' + e.message + '</span>';
                            }}
                            loader.classList.add('hidden');
                        }}
                    }}
                }});
            }});

            // Re-highlight on theme change
            window.addEventListener('themeChanged', async () => {{
                await highlightQuickstart();
                document.querySelectorAll('details[open][data-endpoint]').forEach(async details => {{
                    const id = details.id;
                    const endpoint = details.dataset.endpoint;
                    const dataContainer = document.getElementById('data-' + id);
                    if (dataContainer && dataContainer.innerHTML) {{
                        const response = await fetch(endpoint);
                        const json = await response.json();
                        await highlightJson(json, 'data-' + id);
                        dataContainer.dataset.theme = isDark() ? 'dark' : 'light';
                    }}
                }});
            }});

            const quickstartCode = `curl -X POST http://localhost:8000/v1/chat/completions \\\\
  -H "Content-Type: application/json" \\\\
  -d '{{"model": "claude-sonnet-4-5-20250929", "messages": [{{"role": "user", "content": "Hello!"}}]}}'`;

            async function highlightQuickstart() {{
                const theme = isDark() ? darkTheme : lightTheme;
                try {{
                    const html = await codeToHtml(quickstartCode, {{ lang: 'bash', theme }});
                    document.getElementById('quickstart-code').innerHTML = html;
                }} catch (e) {{
                    document.getElementById('quickstart-code').innerHTML = '<pre>' + quickstartCode + '</pre>';
                }}
            }}

            window.highlightQuickstart = highlightQuickstart;
            highlightQuickstart();
        </script>
        <script>
            const quickstartText = 'curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d \\'{{"model": "claude-sonnet-4-5-20250929", "messages": [{{"role": "user", "content": "Hello!"}}]}}\\'';

            function copyQuickstart() {{
                if (navigator.clipboard && navigator.clipboard.writeText) {{
                    navigator.clipboard.writeText(quickstartText).then(showCopySuccess).catch(fallbackCopy);
                }} else {{
                    fallbackCopy();
                }}
            }}

            function fallbackCopy() {{
                const textarea = document.createElement('textarea');
                textarea.value = quickstartText;
                textarea.style.position = 'fixed';
                textarea.style.opacity = '0';
                document.body.appendChild(textarea);
                textarea.select();
                try {{ document.execCommand('copy'); showCopySuccess(); }} catch (e) {{ console.error('Copy failed:', e); }}
                document.body.removeChild(textarea);
            }}

            function showCopySuccess() {{
                const copyIcon = document.getElementById('copy-icon');
                const checkIcon = document.getElementById('check-icon');
                copyIcon.classList.add('hidden');
                checkIcon.classList.remove('hidden');
                setTimeout(() => {{
                    copyIcon.classList.remove('hidden');
                    checkIcon.classList.add('hidden');
                }}, 2000);
            }}

            function toggleTheme() {{
                const html = document.documentElement;
                const current = html.getAttribute('data-theme');
                const next = current === 'dark' ? 'light' : 'dark';
                html.setAttribute('data-theme', next);
                localStorage.setItem('theme', next);
                updateThemeIcon(next === 'dark');
                window.dispatchEvent(new Event('themeChanged'));
            }}

            function updateThemeIcon(isDark) {{
                document.getElementById('sun-icon').classList.toggle('hidden', isDark);
                document.getElementById('moon-icon').classList.toggle('hidden', !isDark);
            }}

            document.addEventListener('DOMContentLoaded', () => {{
                const saved = localStorage.getItem('theme');
                if (saved) {{
                    document.documentElement.setAttribute('data-theme', saved);
                    updateThemeIcon(saved === 'dark');
                }} else {{
                    updateThemeIcon(true);
                }}
            }});
        </script>
    </head>
    <body>
        <main class="container">
            <!-- Header -->
            <header class="header-flex">
                <div class="header-left">
                    <div class="logo-container">
                        <div class="logo-inner">
                            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"/>
                            </svg>
                        </div>
                    </div>
                    <div>
                        <h1 style="margin:0;">Claude Code OpenAI Wrapper</h1>
                        <p style="margin:0;color:var(--pico-muted-color);">OpenAI-compatible API for Claude</p>
                    </div>
                </div>
                <div class="header-right">
                    <span class="version-badge">v{__version__}</span>
                    <button onclick="toggleTheme()" class="icon-btn" title="Toggle theme">
                        <svg id="sun-icon" class="hidden" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"/>
                        </svg>
                        <svg id="moon-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"/>
                        </svg>
                    </button>
                    <a href="https://github.com/aaronlippold/claude-code-openai-wrapper" target="_blank" rel="noopener noreferrer" class="icon-btn" title="View on GitHub">
                        <svg fill="currentColor" viewBox="0 0 24 24">
                            <path fill-rule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clip-rule="evenodd"/>
                        </svg>
                    </a>
                </div>
            </header>

            <!-- Status Card -->
            <article>
                <div class="status-flex">
                    <div class="status-left">
                        <span class="status-dot" style="background-color: {status_color};"></span>
                        <strong>{status_text}</strong>
                    </div>
                    <span class="auth-badge">Auth: <code class="green-code">{auth_method}</code></span>
                </div>
            </article>

            <!-- Quick Start -->
            <article>
                <div class="section-header">
                    <svg class="section-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
                    <strong>Quick Start</strong>
                </div>
                <div class="quickstart-wrapper">
                    <button onclick="copyQuickstart()" class="copy-btn" title="Copy to clipboard">
                        <svg id="copy-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/>
                        </svg>
                        <svg id="check-icon" class="hidden" fill="none" stroke="currentColor" viewBox="0 0 24 24" style="color:#22c55e;">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/>
                        </svg>
                    </button>
                    <div id="quickstart-code"></div>
                </div>
            </article>

            <!-- API Endpoints -->
            <article>
                <div class="section-header">
                    <svg class="section-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg>
                    <strong>API Endpoints</strong>
                </div>

                <!-- Static POST endpoints -->
                <div class="endpoint-item">
                    <span class="badge badge-post">POST</span>
                    <code>/v1/chat/completions</code>
                    <span class="endpoint-desc">OpenAI-compatible chat</span>
                </div>
                <div class="endpoint-item">
                    <span class="badge badge-post">POST</span>
                    <code>/v1/messages</code>
                    <span class="endpoint-desc">Anthropic-compatible</span>
                </div>

                <!-- Expandable GET endpoints -->
                <details id="models" data-endpoint="/v1/models" name="endpoints">
                    <summary>
                        <span class="badge badge-get">GET</span>
                        <code>/v1/models</code>
                        <span class="endpoint-desc">List models</span>
                    </summary>
                    <div class="content">
                        <small id="loader-models" class="hidden">Loading...</small>
                        <div id="data-models"></div>
                    </div>
                </details>

                <details id="auth" data-endpoint="/v1/auth/status" name="endpoints">
                    <summary>
                        <span class="badge badge-get">GET</span>
                        <code>/v1/auth/status</code>
                        <span class="endpoint-desc">Auth status</span>
                    </summary>
                    <div class="content">
                        <small id="loader-auth" class="hidden">Loading...</small>
                        <div id="data-auth"></div>
                    </div>
                </details>

                <details id="sessions" data-endpoint="/v1/sessions" name="endpoints">
                    <summary>
                        <span class="badge badge-get">GET</span>
                        <code>/v1/sessions</code>
                        <span class="endpoint-desc">Active sessions</span>
                    </summary>
                    <div class="content">
                        <small id="loader-sessions" class="hidden">Loading...</small>
                        <div id="data-sessions"></div>
                    </div>
                </details>

                <details id="health" data-endpoint="/health" name="endpoints">
                    <summary>
                        <span class="badge badge-get">GET</span>
                        <code>/health</code>
                        <span class="endpoint-desc">Health check</span>
                    </summary>
                    <div class="content">
                        <small id="loader-health" class="hidden">Loading...</small>
                        <div id="data-health"></div>
                    </div>
                </details>

                <details id="version" data-endpoint="/version" name="endpoints">
                    <summary>
                        <span class="badge badge-get">GET</span>
                        <code>/version</code>
                        <span class="endpoint-desc">API version</span>
                    </summary>
                    <div class="content">
                        <small id="loader-version" class="hidden">Loading...</small>
                        <div id="data-version"></div>
                    </div>
                </details>
            </article>

            <!-- Configuration -->
            <article>
                <div class="section-header">
                    <svg class="section-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
                    <strong>Configuration</strong>
                </div>
                <p>Set <code>CLAUDE_AUTH_METHOD</code> to choose authentication:</p>
                <div class="config-grid">
                    <div class="config-item">
                        <code class="green-code">cli</code>
                        <p>Claude CLI auth</p>
                    </div>
                    <div class="config-item">
                        <code class="green-code">api_key</code>
                        <p>ANTHROPIC_API_KEY</p>
                    </div>
                    <div class="config-item">
                        <code class="green-code">bedrock</code>
                        <p>AWS Bedrock</p>
                    </div>
                    <div class="config-item">
                        <code class="green-code">vertex</code>
                        <p>Google Vertex AI</p>
                    </div>
                </div>
            </article>

            <!-- Footer -->
            <footer>
                <nav>
                    <a href="/docs">
                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                        </svg>
                        API Docs
                    </a>
                    <a href="/redoc">
                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"/>
                        </svg>
                        ReDoc
                    </a>
                </nav>
            </footer>
        </main>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/v1/debug/request")
@rate_limit_endpoint("debug")
async def debug_request_validation(request: Request):
    """Debug endpoint to test request validation and see what's being sent."""
    try:
        # Get the raw request body
        body = await request.body()
        raw_body = body.decode() if body else ""

        # Try to parse as JSON
        parsed_body = None
        json_error = None
        try:
            import json as json_lib

            parsed_body = json_lib.loads(raw_body) if raw_body else {}
        except Exception as e:
            json_error = str(e)

        # Try to validate against our model
        validation_result = {"valid": False, "errors": []}
        if parsed_body:
            try:
                chat_request = ChatCompletionRequest(**parsed_body)
                validation_result = {"valid": True, "validated_data": chat_request.model_dump()}
            except ValidationError as e:
                validation_result = {
                    "valid": False,
                    "errors": [
                        {
                            "field": " -> ".join(str(loc) for loc in error.get("loc", [])),
                            "message": error.get("msg", "Unknown error"),
                            "type": error.get("type", "validation_error"),
                            "input": error.get("input"),
                        }
                        for error in e.errors()
                    ],
                }

        return {
            "debug_info": {
                "headers": dict(request.headers),
                "method": request.method,
                "url": str(request.url),
                "raw_body": raw_body,
                "json_parse_error": json_error,
                "parsed_body": parsed_body,
                "validation_result": validation_result,
                "debug_mode_enabled": DEBUG_MODE or VERBOSE,
                "example_valid_request": {
                    "model": "claude-3-sonnet-20240229",
                    "messages": [{"role": "user", "content": "Hello, world!"}],
                    "stream": False,
                },
            }
        }

    except Exception as e:
        return {
            "debug_info": {
                "error": f"Debug endpoint error: {str(e)}",
                "headers": dict(request.headers),
                "method": request.method,
                "url": str(request.url),
            }
        }


@app.get("/v1/auth/status")
@rate_limit_endpoint("auth")
async def get_auth_status(request: Request):
    """Get Claude Code authentication status."""
    from src.auth import auth_manager

    auth_info = get_claude_code_auth_info()
    active_api_key = auth_manager.get_api_key()

    return {
        "claude_code_auth": auth_info,
        "server_info": {
            "api_key_required": bool(active_api_key),
            "api_key_source": (
                "environment"
                if os.getenv("API_KEY")
                else ("runtime" if runtime_api_key else "none")
            ),
            "version": "1.0.0",
        },
    }


@app.get("/v1/sessions/stats")
async def get_session_stats(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Get session manager statistics."""
    stats = session_manager.get_stats()
    return {
        "session_stats": stats,
        "cleanup_interval_minutes": session_manager.cleanup_interval_minutes,
        "default_ttl_hours": session_manager.default_ttl_hours,
    }


@app.get("/v1/sessions")
async def list_sessions(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """List all active sessions."""
    sessions = session_manager.list_sessions()
    return SessionListResponse(sessions=sessions, total=len(sessions))


@app.get("/v1/sessions/{session_id}")
async def get_session(
    session_id: str, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """Get information about a specific session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return session.to_session_info()


@app.delete("/v1/sessions/{session_id}")
async def delete_session(
    session_id: str, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """Delete a specific session."""
    deleted = session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"message": f"Session {session_id} deleted successfully"}


# Tool Management Endpoints


@app.get("/v1/tools", response_model=ToolListResponse)
@rate_limit_endpoint("general")
async def list_tools(
    request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """List all available Claude Code tools with metadata."""
    await verify_api_key(request, credentials)

    tools = tool_manager.list_all_tools()
    tool_responses = [
        ToolMetadataResponse(
            name=tool.name,
            description=tool.description,
            category=tool.category,
            parameters=tool.parameters,
            examples=tool.examples,
            is_safe=tool.is_safe,
            requires_network=tool.requires_network,
        )
        for tool in tools
    ]

    return ToolListResponse(tools=tool_responses, total=len(tool_responses))


@app.get("/v1/tools/config", response_model=ToolConfigurationResponse)
@rate_limit_endpoint("general")
async def get_tool_config(
    request: Request,
    session_id: Optional[str] = None,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Get tool configuration (global or per-session)."""
    await verify_api_key(request, credentials)

    config = tool_manager.get_effective_config(session_id)
    effective_tools = tool_manager.get_effective_tools(session_id)

    return ToolConfigurationResponse(
        allowed_tools=config.allowed_tools,
        disallowed_tools=config.disallowed_tools,
        effective_tools=effective_tools,
        created_at=config.created_at,
        updated_at=config.updated_at,
    )


@app.post("/v1/tools/config", response_model=ToolConfigurationResponse)
@rate_limit_endpoint("general")
async def update_tool_config(
    config_request: ToolConfigurationRequest,
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Update tool configuration (global or per-session)."""
    await verify_api_key(request, credentials)

    # Validate tool names if provided
    all_tool_names = []
    if config_request.allowed_tools:
        all_tool_names.extend(config_request.allowed_tools)
    if config_request.disallowed_tools:
        all_tool_names.extend(config_request.disallowed_tools)

    if all_tool_names:
        validation = tool_manager.validate_tools(all_tool_names)
        invalid_tools = [name for name, valid in validation.items() if not valid]
        if invalid_tools:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tool names: {', '.join(invalid_tools)}. Valid tools: {', '.join(CLAUDE_TOOLS)}",
            )

    # Update configuration
    if config_request.session_id:
        config = tool_manager.set_session_config(
            config_request.session_id, config_request.allowed_tools, config_request.disallowed_tools
        )
    else:
        config = tool_manager.update_global_config(
            config_request.allowed_tools, config_request.disallowed_tools
        )

    effective_tools = tool_manager.get_effective_tools(config_request.session_id)

    return ToolConfigurationResponse(
        allowed_tools=config.allowed_tools,
        disallowed_tools=config.disallowed_tools,
        effective_tools=effective_tools,
        created_at=config.created_at,
        updated_at=config.updated_at,
    )


@app.get("/v1/tools/stats")
@rate_limit_endpoint("general")
async def get_tool_stats(
    request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """Get statistics about tool configuration and usage."""
    await verify_api_key(request, credentials)
    return tool_manager.get_stats()


# MCP (Model Context Protocol) Management Endpoints


@app.get("/v1/mcp/servers", response_model=MCPServersListResponse)
@rate_limit_endpoint("general")
async def list_mcp_servers(
    request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """List all registered MCP servers."""
    await verify_api_key(request, credentials)

    if not mcp_client.is_available():
        raise HTTPException(
            status_code=503, detail="MCP SDK not available. Install with: pip install mcp"
        )

    servers = mcp_client.list_servers()
    connections = mcp_client.list_connected_servers()

    server_responses = []
    for server in servers:
        connection = mcp_client.get_connection(server.name)
        server_responses.append(
            MCPServerInfoResponse(
                name=server.name,
                command=server.command,
                args=server.args,
                description=server.description,
                enabled=server.enabled,
                connected=server.name in connections,
                tools_count=len(connection.available_tools) if connection else 0,
                resources_count=len(connection.available_resources) if connection else 0,
                prompts_count=len(connection.available_prompts) if connection else 0,
            )
        )

    return MCPServersListResponse(servers=server_responses, total=len(server_responses))


@app.post("/v1/mcp/servers")
@rate_limit_endpoint("general")
async def register_mcp_server(
    body: MCPServerConfigRequest,
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Register a new MCP server."""
    await verify_api_key(request, credentials)

    if not mcp_client.is_available():
        raise HTTPException(
            status_code=503, detail="MCP SDK not available. Install with: pip install mcp"
        )

    config = MCPServerConfig(
        name=body.name,
        command=body.command,
        args=body.args,
        env=body.env,
        description=body.description,
        enabled=body.enabled,
    )

    mcp_client.register_server(config)

    return {"message": f"MCP server '{body.name}' registered successfully"}


@app.post("/v1/mcp/connect")
@rate_limit_endpoint("general")
async def connect_mcp_server(
    body: MCPConnectionRequest,
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Connect to a registered MCP server."""
    await verify_api_key(request, credentials)

    if not mcp_client.is_available():
        raise HTTPException(
            status_code=503, detail="MCP SDK not available. Install with: pip install mcp"
        )

    success = await mcp_client.connect_server(body.server_name)

    if not success:
        raise HTTPException(
            status_code=500, detail=f"Failed to connect to MCP server '{body.server_name}'"
        )

    connection = mcp_client.get_connection(body.server_name)
    return {
        "message": f"Connected to MCP server '{body.server_name}'",
        "tools": len(connection.available_tools) if connection else 0,
        "resources": len(connection.available_resources) if connection else 0,
        "prompts": len(connection.available_prompts) if connection else 0,
    }


@app.post("/v1/mcp/disconnect")
@rate_limit_endpoint("general")
async def disconnect_mcp_server(
    body: MCPConnectionRequest,
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Disconnect from an MCP server."""
    await verify_api_key(request, credentials)

    if not mcp_client.is_available():
        raise HTTPException(
            status_code=503, detail="MCP SDK not available. Install with: pip install mcp"
        )

    success = await mcp_client.disconnect_server(body.server_name)

    if not success:
        raise HTTPException(
            status_code=404, detail=f"Not connected to MCP server '{body.server_name}'"
        )

    return {"message": f"Disconnected from MCP server '{body.server_name}'"}


@app.get("/v1/mcp/stats")
@rate_limit_endpoint("general")
async def get_mcp_stats(
    request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """Get statistics about MCP connections."""
    await verify_api_key(request, credentials)
    return mcp_client.get_stats()


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Format HTTP exceptions as OpenAI-style errors.

    ``ClaudeProxyError`` carries a specific ``type``/``code`` (e.g.
    context_length_exceeded); plain ``HTTPException`` falls back to the generic
    api_error envelope keyed by status code.
    """
    error_type = getattr(exc, "error_type", "api_error")
    code = getattr(exc, "code", str(exc.status_code))
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": exc.detail, "type": error_type, "code": code}},
    )


def find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    import socket

    for port in range(start_port, start_port + max_attempts):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            result = sock.connect_ex(("127.0.0.1", port))
            if result != 0:  # Port is available
                return port
        except Exception:
            return port
        finally:
            sock.close()

    raise RuntimeError(
        f"No available ports found in range {start_port}-{start_port + max_attempts - 1}"
    )


def run_server(port: int = None, host: str = None):
    """Run the server - used as Poetry script entry point."""
    import uvicorn

    # Handle interactive API key protection
    global runtime_api_key
    runtime_api_key = prompt_for_api_protection()

    # Priority: CLI arg > ENV var > default
    if port is None:
        port = int(os.getenv("PORT", "8000"))
    if host is None:
        # Default to 0.0.0.0 for container/development use (configurable via CLAUDE_WRAPPER_HOST env)
        host = os.getenv("CLAUDE_WRAPPER_HOST", "0.0.0.0")  # nosec B104
    preferred_port = port

    try:
        # Try the preferred port first
        # Binding to 0.0.0.0 is intentional for container/development use
        uvicorn.run(app, host=host, port=preferred_port)  # nosec B104
    except OSError as e:
        if "Address already in use" in str(e) or e.errno == 48:
            logger.warning(f"Port {preferred_port} is already in use. Finding alternative port...")
            try:
                available_port = find_available_port(preferred_port + 1)
                logger.info(f"Starting server on alternative port {available_port}")
                print(f"\n🚀 Server starting on http://localhost:{available_port}")
                print(f"📝 Update your client base_url to: http://localhost:{available_port}/v1")
                # Binding to 0.0.0.0 is intentional for container/development use
                uvicorn.run(app, host=host, port=available_port)  # nosec B104
            except RuntimeError as port_error:
                logger.error(f"Could not find available port: {port_error}")
                print(f"\n❌ Error: {port_error}")
                print("💡 Try setting a specific port with: PORT=9000 poetry run python main.py")
                raise
        else:
            raise


if __name__ == "__main__":
    import sys

    # Simple CLI argument parsing for port
    port = None
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
            print(f"Using port from command line: {port}")
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}. Using default.")

    run_server(port)
