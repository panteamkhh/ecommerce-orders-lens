.PHONY: help install dev run test lint format clean

help:
	@echo "install - install runtime dependencies"
	@echo "dev     - install development dependencies"
	@echo "run     - run the full analysis pipeline"
	@echo "test    - run the test suite"
	@echo "lint    - check lint and formatting"
	@echo "format  - auto-format the code"
	@echo "clean   - remove generated artefacts and caches"

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

run:
	python -m src.run_analysis

test:
	python -m pytest

lint:
	python -m ruff check .
	python -m black --check .

format:
	python -m ruff check . --fix
	python -m black .

clean:
	rm -rf cleaned_data reports .pytest_cache .ruff_cache
