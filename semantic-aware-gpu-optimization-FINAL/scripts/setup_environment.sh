#!/bin/bash

# Setup environment for Semantic-Aware GPU Optimization Framework

set -e

echo "🚀 Setting up environment for Semantic-Aware GPU Optimization"
echo "=============================================================="
echo ""

# Create directories
echo "Creating directories..."
mkdir -p code
mkdir -p data/synthetic
mkdir -p data/results
mkdir -p figures
mkdir -p notebooks
mkdir -p tests
mkdir -p scripts
mkdir -p logs
mkdir -p docs
mkdir -p .github/workflows
mkdir -p .github/ISSUE_TEMPLATE
echo "✓ Directories created"

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r code/requirements.txt
echo "✓ Dependencies installed"

# Install development tools (optional)
echo ""
echo "Installing development tools (optional)..."
pip install pytest pytest-cov black flake8 mypy 2>/dev/null || echo "⚠️  Some dev tools failed (continue anyway)"
echo "✓ Development tools installed"

# Make scripts executable
echo ""
echo "Making scripts executable..."
chmod +x scripts/*.sh
echo "✓ Scripts executable"

# Validate environment
echo ""
echo "Validating environment..."
python scripts/validate_environment.py
echo ""

echo "✅ Environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Review INSTALLATION.md"
echo "2. Run: bash scripts/run_all_experiments.sh"
echo "3. Check: data/results/ and figures/"
echo ""
