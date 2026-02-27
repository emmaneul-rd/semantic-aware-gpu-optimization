# Contributing to Semantic-Aware GPU Optimization

Thank you for your interest in contributing! This document provides guidelines for contributing.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/emmaneul-rd/semantic-aware-gpu-optimization/issues)
2. If not, [open a new issue](https://github.com/emmaneul-rd/semantic-aware-gpu-optimization/issues/new?template=bug_report.md)
3. Provide:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and OS
   - Relevant code/error output

### Suggesting Enhancements

1. [Open a new issue](https://github.com/emmaneul-rd/semantic-aware-gpu-optimization/issues/new?template=feature_request.md)
2. Describe the enhancement and use case
3. Explain why this would be useful to the community

### Code Contributions

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Write tests for new functionality
5. Ensure all tests pass (`pytest tests/ -v`)
6. Commit with clear messages (`git commit -am 'Add feature: ...'`)
7. Push to your fork (`git push origin feature/my-feature`)
8. Open a [Pull Request](https://github.com/emmaneul-rd/semantic-aware-gpu-optimization/pulls)

### Code Style

We follow PEP 8 with some extensions:

- Use `black` for formatting: `black code/ tests/`
- Use `flake8` for linting: `flake8 code/ tests/`
- Use `mypy` for type checking: `mypy code/`
- Line length: 120 characters (not 80)
- Document all functions with docstrings

### Testing Requirements

All new code must include tests:

```python
def test_my_feature():
    """Test description"""
    # Test implementation
    assert result == expected
```

Run tests before submitting:

```bash
pytest tests/ -v --cov=code
```

### Documentation

- Update README.md if relevant
- Document new functions in docstrings
- Add examples if introducing new functionality
- Update CHANGELOG.md

### Pull Request Process

1. Ensure your branch is up-to-date with `main`
2. Fill out the PR template completely
3. Describe what your PR does
4. Reference any related issues (#123)
5. Ensure CI passes (GitHub Actions)
6. Request review from maintainers
7. Address feedback and iterate

### Commit Messages

Use clear, descriptive commit messages:

```
Add: Brief description of what was added

Longer explanation if needed. Explain the "why" not just the "what".
Reference issues: Fixes #123
```

Prefix with:
- `Add:` for new features
- `Fix:` for bug fixes
- `Refactor:` for code reorganization
- `Docs:` for documentation updates
- `Test:` for test additions
- `Chore:` for maintenance tasks

## Development Setup

```bash
# Clone and setup
git clone https://github.com/emmaneul-rd/semantic-aware-gpu-optimization.git
cd semantic-aware-gpu-optimization

# Install development dependencies
pip install -e ".[dev]"

# Pre-commit checks
black code/ tests/
flake8 code/ tests/
mypy code/
pytest tests/ -v
```

## Questions?

- Check [FAQ.md](FAQ.md)
- Open a [Discussion](https://github.com/emmaneul-rd/semantic-aware-gpu-optimization/discussions)
- Contact: emmanuel@salomoncoral.com

## Recognition

Contributors will be recognized in:
- AUTHORS.md
- Commit history
- Release notes

Thank you for contributing! 🎉
