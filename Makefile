.PHONY: install run debug clean lint

PYTHON := uv run python
PROJECT_PATHS := main.py src tests
MYPY_PATHS := main.py src

install:
	uv sync

run:
	$(PYTHON) -m src

debug:
	$(PYTHON) -m pdb -m src

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	find . -type d -name .mypy_cache -prune -exec rm -rf {} +

lint:
	uv run flake8 . --exclude=.venv,llm_sdk --max-line-length=180
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude '(^|/)(\\.venv|llm_sdk|tests|\\.asher)/'
