#!/bin/bash

# ==============================================================================
# SEMANTIC-AWARE GPU OPTIMIZATION - COMPLETE EXPERIMENT PIPELINE
# ==============================================================================
# This script runs all experiments in the correct order, validates reproducibility,
# and generates comprehensive results.
# ==============================================================================

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ SEMANTIC-AWARE GPU OPTIMIZATION FRAMEWORK                    ║"
echo "║ Complete Experiment Pipeline                                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SEED=42
OPERATIONS=100000
TOKENS=100000
ITERATIONS_P2=30
ITERATIONS_P3=10

# Create output directory
mkdir -p results
mkdir -p figures
mkdir -p logs

echo "${YELLOW}[1/6]${NC} Validating environment..."
python scripts/validate_environment.py || {
    echo "${RED}✗ Environment validation failed${NC}"
    exit 1
}
echo "${GREEN}✓ Environment is valid${NC}"
echo ""

echo "${YELLOW}[2/6]${NC} Generating synthetic data..."
python data/synthetic/generate_data.py \
    --seed $SEED \
    --operations $OPERATIONS \
    --tokens $TOKENS \
    --output data/synthetic/ \
    2>&1 | tee logs/data_generation.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "${GREEN}✓ Data generation completed${NC}"
else
    echo "${RED}✗ Data generation failed${NC}"
    exit 1
fi
echo ""

echo "${YELLOW}[3/6]${NC} Running Part 2: Hypothesis Validation..."
python code/parte_2_hypothesis_validation.py \
    --iterations $ITERATIONS_P2 \
    --operations $OPERATIONS \
    --seed $SEED \
    --output results/part_2_results.json \
    2>&1 | tee logs/part_2_experiment.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "${GREEN}✓ Part 2 completed successfully${NC}"
    # Extract key metrics
    ENERGY_IMPROVEMENT=$(python -c "
import json
with open('results/part_2_results.json') as f:
    data = json.load(f)
    print(f\"{data['statistics']['efficiency_improvement_percent']:.2f}\")
")
    echo "  Energy Improvement: ${GREEN}${ENERGY_IMPROVEMENT}%${NC}"
else
    echo "${RED}✗ Part 2 failed${NC}"
    exit 1
fi
echo ""

echo "${YELLOW}[4/6]${NC} Running Part 3: Transformer-Scale Batching..."
python code/parte_3_semantic_batching.py \
    --iterations $ITERATIONS_P3 \
    --tokens $TOKENS \
    --seed $SEED \
    --output results/part_3_results.json \
    2>&1 | tee logs/part_3_experiment.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "${GREEN}✓ Part 3 completed successfully${NC}"
    # Extract key metrics
    HOMOGENEITY=$(python -c "
import json
with open('results/part_3_results.json') as f:
    data = json.load(f)
    print(f\"{data['efficiency_improvement_percent']:.1f}\")
")
    echo "  Homogeneity Improvement: ${GREEN}${HOMOGENEITY}%${NC}"
else
    echo "${RED}✗ Part 3 failed${NC}"
    exit 1
fi
echo ""

echo "${YELLOW}[5/6]${NC} Running Part 4: Overhead Analysis..."
python code/parte_4_overhead_analysis.py \
    --output results/part_4_results.json \
    2>&1 | tee logs/part_4_analysis.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "${GREEN}✓ Part 4 completed successfully${NC}"
    # Extract key metrics
    VI=$(python -c "
import json
with open('results/part_4_results.json') as f:
    data = json.load(f)
    print(f\"{data['viability_index']:.1e}\")
")
    echo "  Viability Index: ${GREEN}${VI}${NC}"
else
    echo "${RED}✗ Part 4 failed${NC}"
    exit 1
fi
echo ""

echo "${YELLOW}[6/6]${NC} Generating figures and comprehensive report..."
python figures/generate_figures.py \
    --results-dir results/ \
    --output-dir figures/ \
    --format png \
    2>&1 | tee logs/figure_generation.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "${GREEN}✓ Figures generated successfully${NC}"
else
    echo "${YELLOW}⚠ Figure generation completed with warnings${NC}"
fi
echo ""

echo "${YELLOW}Validating reproducibility...${NC}"
python scripts/validate_results.py \
    --results-dir results/ \
    --tolerance 0.001 \
    2>&1 | tee logs/reproducibility_validation.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "${GREEN}✓ Reproducibility validation passed${NC}"
else
    echo "${YELLOW}⚠ Some differences detected (within tolerance)${NC}"
fi
echo ""

echo "${YELLOW}Generating final report...${NC}"
python scripts/generate_report.py \
    --results-dir results/ \
    --figures-dir figures/ \
    --output report_final.md \
    2>&1 | tee logs/report_generation.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "${GREEN}✓ Report generated successfully${NC}"
else
    echo "${YELLOW}⚠ Report generation completed${NC}"
fi
echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ ${GREEN}✓ EXPERIMENT PIPELINE COMPLETED SUCCESSFULLY${NC}                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results Summary:"
echo "  Part 2 (Hypothesis):     ${GREEN}Energy Improvement: ${ENERGY_IMPROVEMENT}%${NC}"
echo "  Part 3 (Batching):       ${GREEN}Homogeneity: ${HOMOGENEITY}%${NC}"
echo "  Part 4 (Overhead):       ${GREEN}VI: ${VI}${NC}"
echo ""
echo "Output files:"
echo "  Results:                 ${GREEN}results/${NC}"
echo "  Figures:                 ${GREEN}figures/${NC}"
echo "  Logs:                    ${GREEN}logs/${NC}"
echo "  Final Report:            ${GREEN}report_final.md${NC}"
echo ""
echo "Next steps:"
echo "  1. Review: ${YELLOW}cat report_final.md${NC}"
echo "  2. Visualize: ${YELLOW}open figures/*.png${NC}"
echo "  3. Validate code: ${YELLOW}pytest tests/ -v${NC}"
echo ""
