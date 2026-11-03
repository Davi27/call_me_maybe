.PHONY: setup-storage install run debug clean clean-env lint

# Storage configuration
USER_NAME := $(shell id -un)
PROJECT_NAME := $(notdir $(CURDIR))

SGOINFRE := /sgoinfre/$(USER_NAME)

UV_CACHE_DIR := $(SGOINFRE)/uv-cache
UV_PROJECT_ENVIRONMENT := $(SGOINFRE)/venvs/$(PROJECT_NAME)
HF_HOME := $(SGOINFRE)/huggingface

export UV_CACHE_DIR
export UV_PROJECT_ENVIRONMENT
export HF_HOME

PYTHON := uv run python
PROJECT_PATHS := main.py src tests
MYPY_PATHS := main.py src

# Storage setup
setup-storage:
	mkdir -p "$(UV_CACHE_DIR)"
	mkdir -p "$(SGOINFRE)/venvs"
	mkdir -p "$(HF_HOME)"

# Dependency installation
install: setup-storage
	uv sync

# Execution
run: install
	$(PYTHON) -m src

debug: install
	$(PYTHON) -m pdb -m src

# Linting
lint: install
	uv run flake8 . --exclude=.venv,llm_sdk --max-line-length=180
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude '(^|/)(\\.venv|llm_sdk|tests)/'

# Cleanup
clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	find . -type d -name .mypy_cache -prune -exec rm -rf {} +

clean-env:
	rm -rf "$(UV_PROJECT_ENVIRONMENT)"
