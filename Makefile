.PHONY: dev run test test-cov test-cov-html lint lint-check format install clean build publish publish-all release setup-hooks setup check-python check-poetry check-acceleration help docker-bake docker-bake-prod docker-bake-dev docker-bake-gpu docker-bake-all docker-bake-multiplatform docker-push docker-push-prod docker-push-dev docker-push-gpu docker-push-all docker-push-multiplatform docker-push-tag docker-push-prod-tag docker-clean get-version version-patch version-minor version-major

# Development
dev: install
	@echo "Starting Captiv development server..."
	@poetry run watchfiles "python -m src.captiv.gui" --filter python

run:
	@echo "Starting Captiv..."
	@poetry run python -m src.captiv.gui

# Testing
test:
	@echo "Running tests..."
	@poetry run pytest

test-cov:
	@echo "Running tests with coverage..."
	@poetry run pytest --cov=src/captiv --cov-report=term-missing

test-cov-html:
	@echo "Running tests with HTML coverage..."
	@poetry run pytest --cov=src/captiv --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

# Code Quality
lint:
	@echo "Linting and fixing code..."
	@poetry run ruff check --fix --unsafe-fixes --show-fixes .

lint-check:
	@echo "Checking code without fixing..."
	@poetry run ruff check .

format:
	@echo "Formatting code..."
	@poetry run ruff format .
	@poetry run docformatter --in-place --recursive --black --pre-summary-newline .

# Dependencies
install:
	@echo "Installing dependencies..."
	@poetry install

# Build & Publish
clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +; \
	find . -type f -name "*.pyc" -delete; \
	find . -type f -name "*.pyo" -delete; \
	find . -type d -name "*.egg-info" -exec rm -rf {} +; \
	rm -rf build dist .pytest_cache .mypy_cache htmlcov .coverage

build: clean
	@echo "Building Captiv..."
	@poetry build

# Version Management
get-version:
	@poetry version -s

version-patch:
	@echo "Bumping patch version..."
	@poetry version patch
	@echo "New version: $$(poetry version -s)"

version-minor:
	@echo "Bumping minor version..."
	@poetry version minor
	@echo "New version: $$(poetry version -s)"

version-major:
	@echo "Bumping major version..."
	@poetry version major
	@echo "New version: $$(poetry version -s)"

# Publishing
publish: build
	@echo "Publishing Captiv to PyPI only..."
	@poetry publish

publish-all: build
	@echo "Publishing Captiv to PyPI and Docker Hub..."
	@VERSION=$$(poetry version -s); \
	echo "Publishing version: $$VERSION"; \
	echo "1. Publishing to PyPI..."; \
	poetry publish; \
	echo "2. Building and pushing Docker image with version $$VERSION..."; \
	IMAGE_TAG=$$VERSION docker buildx bake captiv-runpod-release --push; \
	echo "3. Building and pushing Docker image with latest tag..."; \
	IMAGE_TAG=latest docker buildx bake captiv-runpod --push; \
	echo "✅ Published version $$VERSION to both PyPI and Docker Hub!"

# Complete release workflow
release: lint test
	@echo "Starting release workflow..."
	@VERSION_TYPE=$${VERSION:-patch}; \
	echo "1. Bumping $$VERSION_TYPE version..."; \
	poetry version $$VERSION_TYPE; \
	NEW_VERSION=$$(poetry version -s); \
	echo "2. New version: $$NEW_VERSION"; \
	echo "3. Running tests and linting..."; \
	$(MAKE) lint test; \
	echo "4. Building package..."; \
	$(MAKE) build; \
	echo "5. Publishing to PyPI and Docker Hub..."; \
	$(MAKE) publish-all; \
	echo "✅ Release $$NEW_VERSION completed successfully!"

# Setup & Validation
check-python:
	@echo "Checking Python version..."
	@python -c "import sys; version=sys.version_info; exit(0 if version.major == 3 and version.minor >= 12 else 1)" || \
		(echo "Error: Python 3.12 or higher is required. Current version: $$(python --version)" && exit 1)
	@echo "Python version is compatible: $$(python --version)"

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
	@poetry run pre-commit install -t pre-commit -t pre-push

check-acceleration:
	@echo "Checking hardware acceleration..."
	@poetry run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || echo "CUDA check failed"
	@poetry run python -c "import torch; print(f'MPS available: {hasattr(torch, \"mps\") and torch.mps.is_available() if hasattr(torch, \"mps\") else False}')" || echo "MPS check failed"

setup: check-python check-poetry install setup-hooks
	@echo "Setting up development environment..."
	@$(MAKE) -k check-acceleration || true
	@echo "Running linting checks..."
	@$(MAKE) lint
	@echo "Running tests..."
	@$(MAKE) test
	@echo "Development environment setup complete!"

# Help
help:
	@echo "Captiv Makefile Commands:"
	@echo ""
	@echo "Development:"
	@echo "  dev                    - Start development server with file watching"
	@echo "  run                    - Start Captiv GUI"
	@echo "  setup                  - Complete development environment setup"
	@echo ""
	@echo "Testing:"
	@echo "  test                   - Run tests"
	@echo "  test-cov               - Run tests with coverage"
	@echo "  test-cov-html          - Run tests with HTML coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint                   - Lint and fix code"
	@echo "  lint-check             - Check code without fixing"
	@echo "  format                 - Format code"
	@echo ""
	@echo "Build & Publish:"
	@echo "  build                  - Build Python package"
	@echo "  publish                - Publish to PyPI only"
	@echo "  publish-all            - Publish to PyPI and Docker Hub with version sync"
	@echo "  release [VERSION=x]    - Complete release workflow (defaults to patch)"
	@echo ""
	@echo "Version Management:"
	@echo "  get-version            - Get current package version"
	@echo "  version-patch          - Bump patch version (0.1.0 -> 0.1.1)"
	@echo "  version-minor          - Bump minor version (0.1.0 -> 0.2.0)"
	@echo "  version-major          - Bump major version (0.1.0 -> 1.0.0)"
	@echo "  clean                  - Clean build artifacts"
	@echo ""
	@echo "Docker Build:"
	@echo "  docker-bake            - Build RunPod production image"
	@echo "  docker-bake-prod       - Build RunPod production image"
	@echo ""
	@echo "Docker Push:"
	@echo "  docker-push            - Build and push RunPod image"
	@echo "  docker-push-prod       - Build and push RunPod image"
	@echo "  docker-push-tag TAG=x  - Build and push with custom tag"
	@echo ""
	@echo "Docker Maintenance:"
	@echo "  docker-clean           - Clean Docker cache"
	@echo ""
	@echo "Examples:"
	@echo "  make docker-push-tag TAG=v1.2.3"
	@echo "  make publish-all       # Publishes to PyPI and Docker with synced versions"
	@echo "  make release                  # Complete patch release workflow (default)"
	@echo "  make release VERSION=minor    # Complete minor release workflow"
	@echo "  make release VERSION=major    # Complete major release workflow"

# Docker
docker-bake:
	@echo "Building RunPod production image..."
	@docker buildx bake

docker-bake-prod:
	@echo "Building RunPod production image..."
	@docker buildx bake production

# Docker Push Targets
docker-push:
	@echo "Building and pushing RunPod image..."
	@docker buildx bake --push

docker-push-prod:
	@echo "Building and pushing RunPod image..."
	@docker buildx bake production --push

# Docker Push with Custom Tag
docker-push-tag:
	@if [ -z "$(TAG)" ]; then \
		echo "Error: TAG variable is required. Usage: make docker-push-tag TAG=v1.0.0"; \
		exit 1; \
	fi
	@echo "Building and pushing RunPod image with tag: $(TAG)..."
	@IMAGE_TAG=$(TAG) docker buildx bake --push

docker-clean:
	@echo "Cleaning Docker cache..."
	@docker buildx prune -f
	@docker image prune -f
