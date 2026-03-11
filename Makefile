.PHONY: check fix fmt install

check:
	@uvx ruff check && uvx pyrefly check && uvx ty check --ignore invalid-syntax-in-forward-annotation

fix: fmt
	@uvx ruff check --fix

fmt:
	@uvx ruff format

install: .venv
	uv tool install ruff
	uv tool install pyrefly
	uv tool install ty
	uv tool install mypy

.venv:
	(uv venv && . .venv/bin/activate && uv pip install .)
