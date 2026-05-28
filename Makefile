UV := uv

HF_CACHE := /goinfre/$(user)/hf_cache
UV_CACHE_DIR := /goinfre/$(user)/uv_cache

install-home:
	uv sync

run-home:
	uv run python -m src

install:
	@mkdir -p $(UV_CACHE_DIR)
	@UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) sync

run:
	@mkdir -p $(HF_CACHE)
	@HF_HOME=$(HF_CACHE)
	HUGGINGFASE_HUB_CACHE=$(HF_CACHE) \
	TRANSFORMERS_CACHE=$(HF_CACHE) \
	$(UV) run python -m src

debug:
	@mkdir -p $(HF_CACHE)
	@HF_HOME=$(HF_CACHE)
	HUGGINGFASE_HUB_CACHE=$(HF_CACHE) \
	TRANSFORMERS_CACHE=$(HF_CACHE) \
	$(UV) run python -m pdb src

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .mypy_cache

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict