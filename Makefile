.PHONY: setup-storage install run debug clean clean-env lint

# Project paths
PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

# Storage configuration
USER_NAME := $(shell id -un)
SGOINFRE := /sgoinfre/$(USER_NAME)

UV_CACHE_DIR := $(SGOINFRE)/uv-cache
UV_PROJECT_ENVIRONMENT := $(SGOINFRE)/venvs/call_me_maybe
HF_HOME := $(SGOINFRE)/huggingface
TMPDIR := $(SGOINFRE)/tmp

export UV_CACHE_DIR
export UV_PROJECT_ENVIRONMENT
export HF_HOME
export TMPDIR

UV := uv --project "$(PROJECT_DIR)"
PYTHON := $(UV) run python

# Storage setup
setup-storage:
	mkdir -p "$(UV_CACHE_DIR)"
	mkdir -p "$(SGOINFRE)/venvs"
	mkdir -p "$(HF_HOME)"
	mkdir -p "$(TMPDIR)"

# Dependency installation
install: setup-storage
	$(UV) sync

# Execution
run: install
	$(PYTHON) -m src

debug: install
	$(PYTHON) -m pdb -m src

# Linting
lint: install
	$(UV) run flake8 . --exclude=.venv,llm_sdk --max-line-length=180
	$(UV) run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude '(^|/)(\\.venv|llm_sdk|tests)/'

# Cleanup
clean:
	find "$(PROJECT_DIR)" -type d -name __pycache__ -prune -exec rm -rf {} +
	find "$(PROJECT_DIR)" -type d -name .pytest_cache -prune -exec rm -rf {} +
	find "$(PROJECT_DIR)" -type d -name .mypy_cache -prune -exec rm -rf {} +

clean-env:
	rm -rf "$(UV_PROJECT_ENVIRONMENT)"