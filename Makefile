.PHONY: lint format typecheck security test check

lint:
	uv run ruff check src
	uv run ruff format --check src

format:
	uv run ruff format src
	uv run ruff check --fix src

typecheck:
	uv run mypy src

security:
	uv run bandit -r src/ -ll
	uv run pip-audit

test:
	uv run pytest

check: test lint typecheck security
