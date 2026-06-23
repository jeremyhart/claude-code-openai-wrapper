# Developer convenience targets that mirror the CI pipeline in
# .github/workflows/ci.yml. Everything runs through Poetry so the *pinned* tool
# versions (e.g. black 24.10.0) are used -- the exact versions CI enforces.
#
# Quick start:
#     make install   # one-time: deps + format-on-commit git hook
#     make format    # auto-fix formatting before you commit
#     make check     # reproduce the CI gate locally

.DEFAULT_GOAL := help
.PHONY: help install format lint typecheck security test check

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Install pinned dependencies and the format-on-commit git hook
	poetry install
	poetry run pre-commit install

format:  ## Auto-format src and tests with the pinned black (24.10.0)
	poetry run black src tests

lint:  ## Check formatting exactly as CI does -- hard fails if not formatted
	poetry run black --check src tests

typecheck:  ## Run mypy (non-blocking in CI; strict here for local use)
	poetry run mypy src --ignore-missing-imports

security:  ## Run the bandit security scan (hard fails on medium+ issues)
	poetry run bandit -r src/ -ll -x tests

test:  ## Run the test suite with coverage, as CI does
	poetry run pytest tests/ -v --cov=src --cov-report=term-missing

check: lint security test  ## Reproduce CI's blocking gates: black, bandit, pytest (+ mypy)
	-poetry run mypy src --ignore-missing-imports
	@echo "All blocking CI gates passed (black --check, bandit, pytest)."
