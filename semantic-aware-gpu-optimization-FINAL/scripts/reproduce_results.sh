#!/bin/bash

# Reproduce exact results from paper

set -e

echo "🔄 Reproducing Semantic-Aware GPU Optimization Results"
echo "=============================================================="
echo ""

# Check environment
echo "Validating environment..."
python scripts/validate_environment.py || exit 1

# Create directories
mkdir -p data/results data/synthetic

echo ""
echo "Running Part 2: Hypothesis Validation"
python code/parte_2_hypothesis_validation.py

echo ""
echo "Running Part 3: Transformer Batching"
python code/parte_3_semantic_batching.py

echo ""
echo "Running Part 4: Overhead Analysis"
python code/parte_4_overhead_analysis.py

echo ""
echo "Generating figures..."
python figures/generate_figures.py

echo ""
echo "✅ Reproduction complete!"
echo ""
echo "Results saved to:"
echo "  - data/results/"
echo "  - figures/"
echo ""
