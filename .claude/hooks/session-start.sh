#!/bin/bash
#
# SessionStart hook for claude-code-openai-wrapper.
#
# Goal: make "format before commit" the default for Claude Code sessions so
# agents never push unformatted code (which hard-fails the CI `black --check`
# gate). This runs synchronously on every session start.
#
# In remote sessions (Claude Code on the web) the container is provisioned
# fresh, so we install the *pinned* toolchain and wire up the git hook:
#   - `poetry install` makes `poetry run black` resolve to the version pinned in
#     poetry.lock (24.10.0) -- the exact version CI uses.
#   - `pre-commit install` wires black into the git pre-commit hook so it runs
#     automatically on every `git commit`.
#
# In all sessions we inject a short reminder (additionalContext) so the agent
# knows to format with the pinned black before committing.
#
# Idempotent and non-interactive: safe to run on every session start.
set -uo pipefail

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

log() { echo "session-start: $*" >&2; }

# Only do the (potentially slow) dependency install in remote/web sessions,
# where the container is fresh and the result is cached. Local contributors
# manage their own environment via `make install` / `pre-commit install`.
if [ "${CLAUDE_CODE_REMOTE:-}" = "true" ]; then
  if command -v poetry >/dev/null 2>&1; then
    log "installing pinned dependencies (poetry install)..."
    if poetry install --no-interaction >&2; then
      log "wiring up format-on-commit (pre-commit install)..."
      poetry run pre-commit install >&2 2>&1 \
        || log "pre-commit install failed (non-fatal)."
    else
      log "poetry install failed; 'poetry run black' may be unavailable."
    fi
  else
    log "poetry not found on PATH; skipping toolchain setup."
  fi
fi

# Inject a durable reminder into the session context. This makes formatting the
# default behavior for the agent instead of something it has to remember.
cat <<'JSON'
{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"This repo enforces black formatting as a HARD-FAIL CI gate (black is pinned to 24.10.0 with line-length 100 in pyproject.toml/poetry.lock). ALWAYS format before committing: run `poetry run black src tests` (the pinned 24.10.0 -- do NOT use a system `black`, which may be a different version and produce a diff CI rejects). A pre-commit hook also runs black automatically on `git commit`. Never commit code that `poetry run black --check src tests` would reformat. To reproduce CI locally: `make lint` (or `make format` to fix)."}}
JSON
