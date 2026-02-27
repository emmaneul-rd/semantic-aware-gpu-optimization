# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip or conda
- ~500 MB disk space for code and data

## Quick Install (Recommended)

### Using pip

```bash
# Clone the repository
git clone https://github.com/emmaneul-rd/semantic-aware-gpu-optimization.git
cd semantic-aware-gpu-optimization

# Install dependencies
pip install -r code/requirements.txt

# Install as package (development mode)
pip install -e .
```

### Using conda

```bash
# Clone the repository
git clone https://github.com/emmaneul-rd/semantic-aware-gpu-optimization.git
cd semantic-aware-gpu-optimization

# Create environment
conda create -n semantic-gpu python=3.11
conda activate semantic-gpu

# Install dependencies
conda install numpy scipy matplotlib

# Install package
pip install -e .
```

## Verify Installation

```bash
# Check Python version
python --version  # Should be 3.8+

# Check NumPy installation
python -c "import numpy; print(f'NumPy {numpy.__version__}')"

# Run simple test
python -c "from code.parte_2_hypothesis_validation import SemanticCoherenceIndex; print('✓ Installation successful')"
```

## Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install Jupyter for notebooks
pip install -e ".[notebooks]"

# Install all (dev + notebooks)
pip install -e ".[dev,notebooks]"
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=code --cov-report=html

# Run specific test
pytest tests/test_semantic_coherence.py -v
```

## Validate Environment

```bash
# Check all dependencies and environment
python scripts/validate_environment.py

# Run quick reproducibility check
bash scripts/reproduce_results.sh
```

## Troubleshooting

### NumPy Import Error
```bash
# Reinstall NumPy
pip install --upgrade numpy scipy
```

### Permission Denied on Linux/Mac
```bash
# Make scripts executable
chmod +x scripts/*.sh
```

### CUDA/GPU Not Found
This is normal - the code runs on CPU with simulation of GPU behavior. For actual GPU testing, see Phase 2 documentation.

## Next Steps

After installation:
1. Read [USAGE.md](USAGE.md) for how to run experiments
2. Check [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for validation
3. Review [METHODOLOGY.md](METHODOLOGY.md) for technical details

## Getting Help

- Check [FAQ.md](FAQ.md) for common questions
- Open an [Issue](https://github.com/emmaneul-rd/semantic-aware-gpu-optimization/issues)
- Visit [Discussions](https://github.com/emmaneul-rd/semantic-aware-gpu-optimization/discussions)
