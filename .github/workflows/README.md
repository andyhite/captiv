# GitHub Workflows

## CI Workflow (`ci.yml`)

This workflow runs on every push to the `main` branch and on pull requests targeting `main`.

### Jobs

#### Lint Job

- Runs on Ubuntu with Python 3.12
- Uses Poetry for dependency management
- Performs code linting with `ruff`
- Checks code formatting with `ruff format --check`

#### Test Job

- Runs on Ubuntu with Python 3.12 and 3.13 (matrix strategy)
- Uses Poetry for dependency management
- Runs the full test suite with coverage using `pytest`
- Generates coverage reports in XML and terminal formats
- Generates JUnit XML test results for test analytics
- Uploads coverage to Codecov (Python 3.12 only)
- Uploads test results to Codecov for test analytics (Python 3.12 only)
- Adds coverage comments to pull requests (Python 3.12 only)

### Coverage and Test Analytics

The workflow includes comprehensive reporting features:

1. **Terminal Coverage**: Shows coverage summary in the workflow logs
2. **Codecov Integration**: Uploads coverage data to Codecov for tracking over time
3. **Test Analytics**: Uploads JUnit XML test results to Codecov for:
   - Test execution time tracking
   - Test failure analysis
   - Test performance trends
   - Flaky test detection
4. **PR Comments**: Automatically adds coverage comments to pull requests with:
   - Minimum green threshold: 80%
   - Minimum orange threshold: 70%

### Dependencies

The workflow automatically:

- Caches Poetry virtual environments for faster builds
- Installs all project dependencies including dev dependencies
- Uses the latest versions of GitHub Actions

### Requirements

- Python 3.12+ (as specified in `pyproject.toml`)
- Poetry for dependency management
- All dependencies defined in `pyproject.toml`
