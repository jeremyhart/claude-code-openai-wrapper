# ---- Builder stage: install dependencies into an isolated venv ----
FROM python:3.12-slim AS builder

# curl is only needed to fetch the Poetry installer; it stays in this stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:${PATH}" \
    POETRY_NO_INTERACTION=1 \
    # Build an in-project .venv we can copy wholesale into the runtime stage.
    POETRY_VIRTUALENVS_IN_PROJECT=true

WORKDIR /app

# Dependency layer: copy only the manifests so this expensive install is
# cached and only re-runs when pyproject.toml/poetry.lock actually change.
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-directory --without dev \
    && rm -rf "$(poetry config cache-dir)" /root/.cache

# ---- Runtime stage: clean slim base + the venv + source only ----
FROM python:3.12-slim AS runtime

# Note: Claude Code CLI is bundled with claude-agent-sdk >= 0.1.8 (inside the
# venv site-packages), so no separate Node.js/npm install is required.

WORKDIR /app

# Copy just the resolved dependencies. Poetry, curl, and apt/build caches are
# left behind in the builder and never ship in the final image.
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# Source layer: changing app code does not invalidate the dependency layer.
COPY . .

# Expose the port (default 8000)
EXPOSE 8000

# Run the app with Uvicorn (production: no --reload). uvicorn is on PATH via
# the copied venv, so no `poetry run` wrapper is needed.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
