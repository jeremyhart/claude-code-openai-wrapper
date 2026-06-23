FROM python:3.12-slim

# Install system deps (curl for the Poetry installer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry globally and configure it for container use:
#   - no interactive prompts
#   - install into the system environment (no nested virtualenv)
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:${PATH}" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

# Note: Claude Code CLI is bundled with claude-agent-sdk >= 0.1.8
# No separate Node.js/npm installation required

WORKDIR /app

# Dependency layer: copy only the manifests so this expensive install is
# cached and only re-runs when pyproject.toml/poetry.lock actually change.
# Source edits below this line reuse the cached dependencies.
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-directory --without dev

# Source layer: changing app code does not invalidate the dependency layer.
COPY . .

# Expose the port (default 8000)
EXPOSE 8000

# Run the app with Uvicorn (production: no --reload).
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
