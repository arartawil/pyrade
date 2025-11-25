# PyPI Publishing Guide for PyRADE

This guide explains how to publish PyRADE to PyPI so users can install it via `pip install pyrade`.

## Prerequisites

1. **Create PyPI Account**
   - Go to https://pypi.org/account/register/
   - Verify your email address

2. **Create TestPyPI Account (Optional but Recommended)**
   - Go to https://test.pypi.org/account/register/
   - This allows you to test the upload process first

3. **Install Required Tools**
   ```bash
   pip install build twine
   ```

## Step-by-Step Publishing Process

### Step 1: Prepare Your Package

Make sure all files are committed:
```bash
git add .
git commit -m "chore: Prepare for PyPI publication"
git push
```

### Step 2: Build Distribution Packages

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build source distribution and wheel
python -m build
```

This creates:
- `dist/pyrade-0.1.0.tar.gz` (source distribution)
- `dist/pyrade-0.1.0-py3-none-any.whl` (wheel distribution)

### Step 3: Upload to TestPyPI (Recommended First)

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*
```

You'll be prompted for:
- Username: Your TestPyPI username
- Password: Your TestPyPI password

### Step 4: Test Installation from TestPyPI

```bash
# Try installing from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pyrade
```

Test that it works:
```python
from pyrade import DifferentialEvolution
print("PyRADE imported successfully!")
```

### Step 5: Upload to PyPI (Production)

Once testing is successful:

```bash
# Upload to PyPI
python -m twine upload dist/*
```

You'll be prompted for:
- Username: Your PyPI username
- Password: Your PyPI password

### Step 6: Verify Installation

```bash
# Uninstall test version
pip uninstall pyrade

# Install from PyPI
pip install pyrade
```

## Using API Tokens (Recommended)

Instead of entering your password each time, use API tokens:

### For PyPI:
1. Go to https://pypi.org/manage/account/token/
2. Create a new API token (scope: entire account or specific project)
3. Create `~/.pypirc` file:

```ini
[pypi]
username = __token__
password = pypi-your-token-here
```

### For TestPyPI:
1. Go to https://test.pypi.org/manage/account/token/
2. Create token and add to `~/.pypirc`:

```ini
[testpypi]
username = __token__
password = pypi-your-token-here
repository = https://test.pypi.org/legacy/
```

## Automated Publishing with GitHub Actions

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.x'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    - name: Build package
      run: python -m build
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: python -m twine upload dist/*
```

Add your PyPI token to GitHub secrets:
1. Go to your repo settings → Secrets and variables → Actions
2. Add new secret: `PYPI_API_TOKEN` with your PyPI token

## Quick Command Reference

```bash
# Build package
python -m build

# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Upload to PyPI
python -m twine upload dist/*

# Check package validity
python -m twine check dist/*
```

## Version Management

When releasing new versions:

1. Update version in `setup.py` and `pyproject.toml`
2. Create git tag:
   ```bash
   git tag -a v0.2.0 -m "Release v0.2.0"
   git push origin v0.2.0
   ```
3. Create GitHub release
4. Build and upload to PyPI

## Troubleshooting

**Package name already exists:**
- The name "pyrade" might be taken. Check https://pypi.org/project/pyrade/
- Consider alternative names like "pyrade-opt", "pyrade-de", etc.

**Upload fails:**
- Ensure you've built fresh distributions: `python -m build`
- Check package metadata: `python -m twine check dist/*`
- Verify version number is unique (can't re-upload same version)

**Import errors after install:**
- Check `__init__.py` exports are correct
- Verify package structure with `python -m build --check`

## After Publishing

Update README.md installation section:

```markdown
## Installation

```bash
pip install pyrade
```
```

Congratulations! 🎉 PyRADE is now available on PyPI!
