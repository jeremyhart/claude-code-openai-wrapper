"""
Pytest configuration and fixtures for claude-code-openai-wrapper tests.
"""

import os

# Disable the per-endpoint rate limiter for the whole test session. Several
# TestClient-based suites (multiple responses, function calling, enhanced
# streaming) each POST to /v1/chat/completions; with the default 10/minute chat
# limit their combined calls would trip a 429 purely as a function of test
# ordering. This must be set before src.main / src.rate_limiter are imported so
# the global limiter is created as a no-op. Honor an explicit override if the
# operator already set it.
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")

import pytest
import requests


# Check if server is running for integration tests
def is_server_running(base_url: str = "http://localhost:8000") -> bool:
    """Check if the test server is running."""
    try:
        response = requests.get(f"{base_url}/health", timeout=2)
        return response.status_code == 200
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False


# Marker for tests that require a running server
requires_server = pytest.mark.skipif(
    not is_server_running(),
    reason="Server not running at localhost:8000. Start with: poetry run python main.py",
)
