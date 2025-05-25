# Release Guide for Captiv

This guide covers how to publish Captiv to both PyPI and Docker Hub with synchronized versions.

## Prerequisites

### PyPI Authentication

1. Create a PyPI account at <https://pypi.org/account/register/>
2. Generate an API token at <https://pypi.org/manage/account/token/>
3. Configure Poetry with your token:

   ```bash
   poetry config pypi-token.pypi pypi-YOUR_TOKEN_HERE
   ```

### Docker Hub Authentication

1. Create a Docker Hub account at <https://hub.docker.com/>
2. Login to Docker:

   ```bash
   docker login
   ```

## Release Commands

### Quick Release (Recommended)

Use the automated release workflow that handles version bumping, testing, building, and publishing:

```bash
# Patch release (0.1.0 -> 0.1.1) - DEFAULT
make release

# Minor release (0.1.0 -> 0.2.0)
make release VERSION=minor

# Major release (0.1.0 -> 1.0.0)
make release VERSION=major
```

### Manual Release Steps

#### 1. Version Management

```bash
# Check current version
make get-version

# Bump version manually
make version-patch    # 0.1.0 -> 0.1.1
make version-minor    # 0.1.0 -> 0.2.0
make version-major    # 0.1.0 -> 1.0.0
```

#### 2. Quality Checks

```bash
# Run linting and tests
make lint
make test
```

#### 3. Publishing Options

**PyPI Only:**

```bash
make publish
```

**PyPI + Docker Hub (Synchronized):**

```bash
make publish-all
```

This will:

- Publish the Python package to PyPI
- Build and push Docker image with the same version tag
- Build and push Docker image with `latest` tag

## Docker Image Tags

When using `make publish-all`, the following Docker tags are created:

- `andyhite/captiv:X.Y.Z` (version-specific)
- `andyhite/captiv:X.Y.Z-runpod` (RunPod-specific version)
- `andyhite/captiv:latest` (latest version)
- `andyhite/captiv:runpod` (latest RunPod version)
- `andyhite/captiv:production` (production alias)

## Release Workflow

The complete release process:

1. **Development** - Make your changes
2. **Testing** - Run `make test` and `make lint`
3. **Version Bump** - Use `make release` (defaults to patch) or `make release VERSION=minor|major`
4. **Automated Process**:
   - Bumps version in `pyproject.toml`
   - Runs linting and tests
   - Builds Python package
   - Publishes to PyPI
   - Builds and pushes Docker images with matching version tags

## Version Strategy

Follow semantic versioning:

- **Patch** (0.1.0 -> 0.1.1): Bug fixes, small improvements
- **Minor** (0.1.0 -> 0.2.0): New features, backwards compatible
- **Major** (0.1.0 -> 1.0.0): Breaking changes

## Verification

After release, verify:

1. **PyPI Package**: <https://pypi.org/project/captiv/>

   ```bash
   pip install captiv==X.Y.Z
   captiv --help
   ```

2. **Docker Image**: <https://hub.docker.com/r/andyhite/captiv>

   ```bash
   docker pull andyhite/captiv:X.Y.Z
   docker run --rm andyhite/captiv:X.Y.Z captiv --help
   ```

## Troubleshooting

### PyPI Authentication Issues

```bash
# Re-configure token
poetry config pypi-token.pypi pypi-YOUR_NEW_TOKEN

# Test authentication
poetry publish --dry-run
```

### Docker Build Issues

```bash
# Clean Docker cache
make docker-clean

# Test build locally
make docker-bake
```

### Version Conflicts

```bash
# Check current version
make get-version

# Manually set version if needed
poetry version X.Y.Z
```

## Examples

```bash
# Complete patch release (default)
make release

# Complete minor release
make release VERSION=minor

# Manual workflow
make version-minor
make lint
make test
make publish-all

# Docker-only release with custom tag
make docker-push-tag TAG=v1.2.3-beta
