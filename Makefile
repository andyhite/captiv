.PHONY: dev run test lint format install clean build publish

dev: install
	poetry run watchfiles "python -m src.captiv.gui" --filter python

run:
	poetry run python -m src.captiv.gui

test:
	poetry run pytest

lint:
	poetry run ruff check .

format:
	poetry run autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place --exclude=__init__.py .
	poetry run isort .
	poetry run ruff format .

install:
	poetry install

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +; \
	find . -type f -name "*.pyc" -delete; \
	find . -type f -name "*.pyo" -delete; \
	find . -type d -name "*.egg-info" -exec rm -rf {} +; \
	rm -rf build dist .pytest_cache .mypy_cache

build: clean
	poetry build

publish: build
	poetry publish
