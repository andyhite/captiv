.PHONY: dev run test lint format install clean build publish setup-hooks setup check-python check-poetry check-acceleration

dev: install
	@echo "Starting Captiv development server..."
	@poetry run watchfiles "python -m src.captiv.gui" --filter python

run:
	@echo "Starting Captiv..."
	@poetry run python -m src.captiv.gui

test:
	@echo "Running tests..."
	@poetry run pytest

lint:
	@echo "Linting code..."
	@poetry run ruff check .

format:
	@echo "Formatting code..."
	@poetry run autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place --exclude=__init__.py .
	@poetry run isort .
	@poetry run ruff format .

install:
	@echo "Installing dependencies..."
	@poetry install

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +; \
	find . -type f -name "*.pyc" -delete; \
	find . -type f -name "*.pyo" -delete; \
	find . -type d -name "*.egg-info" -exec rm -rf {} +; \
	rm -rf build dist .pytest_cache .mypy_cache

build: clean
	@echo "Building Captiv..."
	@poetry build

publish: build
	@echo "Publishing Captiv..."
	@poetry publish

# Check if Python version is 3.12 or higher
check-python:
	@echo "Checking Python version..."
	@python -c "import sys; version=sys.version_info; exit(0 if version.major == 3 and version.minor >= 12 else 1)" || \
		(echo "Error: Python 3.12 or higher is required. Current version: $$(python --version)" && exit 1)
	@echo "Python version is compatible: $$(python --version)"

# Check if Poetry is installed, install if not
check-poetry:
	@echo "Checking Poetry installation..."
	@if ! command -v poetry &> /dev/null; then \
		echo "Poetry not found. Installing Poetry..."; \
		curl -sSL https://install.python-poetry.org | python3 -; \
	else \
		echo "Poetry is already installed: $$(poetry --version)"; \
	fi

setup-hooks: install
	@echo "Setting up pre-commit hooks..."
	poetry run pre-commit install -t pre-commit -t pre-push

# Check hardware acceleration availability (optional)
check-acceleration:
	@echo "Checking hardware acceleration availability..."
	@poetry run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || echo "PyTorch CUDA check failed - CUDA unavailable or PyTorch installation issue"
	@poetry run python -c "import torch; print(f'MPS available: {hasattr(torch, \"mps\") and torch.mps.is_available() if hasattr(torch, \"mps\") else False}')" || echo "PyTorch MPS check failed - MPS unavailable or PyTorch installation issue"

setup: check-python check-poetry install setup-hooks
	@echo "Setting up development environment..."
	@$(MAKE) -k check-acceleration || true
	@echo "Running linting checks..."
	@make lint
	@echo "Running tests..."
	@make test
	@echo "Development environment setup complete!"
