UV := uv
USER := $(shell whoami)

GOINFRE := /home/$(USER)/sgoinfre/
VENV_DIR := $(GOINFRE)/venvs/call_me_maybe
UV_CACHE_DIR := $(GOINFRE)/uv_cache
HF_CACHE := $(GOINFRE)/hf_cache
MYPY_CACHE_DIR := $(GOINFRE)/mypy_cache

export UV_PROJECT_ENVIRONMENT := $(VENV_DIR)
export UV_CACHE_DIR := $(UV_CACHE_DIR)
export UV_LINK_MODE := copy
export MYPY_CACHE_DIR

export HF_HOME := $(HF_CACHE)
export HUGGINGFACE_HUB_CACHE := $(HF_CACHE)
export TRANSFORMERS_CACHE := $(HF_CACHE)
export MYPY_CACHE_DIR := $(MYPY_CACHE_DIR)

PYTHON := $(UV) run python

.PHONY: setup

setup:
	@mkdir -p $(VENV_DIR)
	@mkdir -p $(UV_CACHE_DIR)
	@mkdir -p $(HF_CACHE)
	@mkdir -p $(MYPY_CACHE_DIR)

.PHONY: install reinstall

install: setup
	$(UV) sync

reinstall: fclean install

.PHONY: run debug

run: setup
	$(PYTHON) -m src

debug: setup
	$(PYTHON) -m pdb src

.PHONY: lint lint-strict

lint: setup
	$(PYTHON) -m flake8 src
	$(PYTHON) -m mypy src

lint-strict: setup
	$(PYTHON) -m flake8 src
	$(PYTHON) -m mypy src --strict

.PHONY: clean fclean

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -rf .mypy_cache
	rm -rf src/*.egg-info

fclean: clean
	rm -rf $(VENV_DIR)
	rm -rf $(UV_CACHE_DIR)
	rm -rf $(MYPY_CACHE_DIR)
	rm -rf $(HF_CACHE)

.PHONY: info

info:
	@echo "USER      = $(USER)"
	@echo "VENV      = $(VENV_DIR)"
	@echo "UV CACHE  = $(UV_CACHE_DIR)"
	@echo "HF CACHE  = $(HF_CACHE)"