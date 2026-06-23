# Contributing

Thanks for contributing! This guide covers the local setup that keeps CI green.

## One-time setup

```bash
make install
# equivalent to:
#   poetry install                 # installs the pinned toolchain (incl. black 24.10.0)
#   poetry run pre-commit install  # wires black into your git commit hook
```

Running `pre-commit install` is important: it installs a git pre-commit hook so
**black runs automatically on every `git commit`**. If your staged changes
aren't formatted, the commit is blocked and the files are reformatted in place —
just `git add` them again and re-commit. This is what stops unformatted code
from reaching a PR (and from breaking `main`).

## Formatting

This project enforces [**black**](https://black.readthedocs.io/) as a
**hard-fail CI check**. Black is pinned to `24.10.0` in `poetry.lock` and
configured with `line-length = 100` under `[tool.black]` in `pyproject.toml`.

- Format your code: `make format` (runs `poetry run black src tests`)
- Check formatting like CI does: `make lint` (runs `poetry run black --check src tests`)

> **Always use the pinned black via Poetry / pre-commit — not a system-wide
> `black`.** Black's formatting changes between releases, so a different version
> can produce a diff that the pinned CI check (`black --check`) rejects. The
> pre-commit hook pins black to the same `24.10.0`, so it stays in sync no matter
> what's on your PATH.

## Reproducing CI locally

CI (`.github/workflows/ci.yml`) hard-fails on formatting, security, and tests.
Reproduce all of it with one command before pushing:

```bash
make check   # black --check + bandit + pytest (blocking, like CI) + mypy (non-blocking)
```

Individual targets are also available — run `make help` to list them
(`format`, `lint`, `typecheck`, `security`, `test`, `check`).

## Using Claude Code on this repo

A `SessionStart` hook (`.claude/`) sets up the pinned toolchain and the
format-on-commit hook automatically for Claude Code sessions, and reminds the
agent to run `poetry run black src tests` before committing. Agents should never
commit code that `poetry run black --check src tests` would reformat.
