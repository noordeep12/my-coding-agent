.PHONY: lint format typecheck security test check

lint:
	uv run ruff check src agents workflows
	uv run ruff format --check src agents workflows

format:
	uv run ruff format src agents workflows
	uv run ruff check --fix src agents workflows

typecheck:
	uv run mypy src

security:
	uv run bandit -r src/ -ll
	uv run pip-audit

test:
	uv run pytest

check: test lint typecheck security
