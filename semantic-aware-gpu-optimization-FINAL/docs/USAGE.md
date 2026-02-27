# Usage Guide

## Quick Start (5 minutes)

### Run All Experiments

```bash
bash scripts/run_all_experiments.sh
```

This will:
1. Generate 100K synthetic operations
2. Run Part 2 (Hypothesis Validation - 30 iterations)
3. Run Part 3 (Transformer Batching - 10 iterations)
4. Run Part 4 (Overhead Analysis)
5. Generate publication-quality figures
6. Validate reproducibility

**Output:** Results in `results/`, figures in `figures/`

## Individual Experiments

### Part 2: Hypothesis Validation

```bash
python code/parte_2_hypothesis_validation.py
```

**What it does:**
- Generates 100K synthetic GPU operations
- Compares random grouping (baseline) vs semantic grouping (optimized)
- Measures energy consumption and cache miss rates
- Performs statistical validation (t-tests, p-values)

**Output:** `data/results/part_2_results.json`

**Example:**
```json
{
  "energy": {
    "random_grouping_gj": 23818115.62,
    "semantic_grouping_gj": 4214230.31,
    "improvement_percent": 82.31
  },
  "statistics": {
    "energy_t_test": {
      "p_value": 2.34e-154,
      "significant": true
    }
  }
}
```

### Part 3: Transformer Batching

```bash
python code/parte_3_semantic_batching.py
```

**What it does:**
- Generates 100K realistic Transformer tokens
- Compares random batching vs semantic batching
- Measures semantic coherence index (σ)
- Validates with actual Transformer workload

**Output:** `data/results/part_3_results.json`

**Example:**
```json
{
  "homogeneity": {
    "random_batching": {
      "mean": 0.0459
    },
    "semantic_batching": {
      "mean": 1.0000
    }
  },
  "statistics": {
    "improvement_percent": 2077.0
  }
}
```

### Part 4: Overhead Analysis

```bash
python code/parte_4_overhead_analysis.py
```

**What it does:**
- Analyzes cost of semantic classification
- Computes Viability Index (VI)
- Proves O(n log n) scalability
- Projects overhead on H100

**Output:** `data/results/part_4_results.json`

**Example:**
```json
{
  "viability": {
    "viability_index": 713881904.6,
    "overhead_as_percent_of_benefit": 0.00000014,
    "viable": true
  }
}
```

## Benchmarking

### Run GPU Benchmarks

```bash
python code/benchmark_simulation.py
```

**What it does:**
- Simulates H100/A100/L40S performance
- Tests multiple scales (10K - 10M operations)
- Measures throughput, energy, memory efficiency
- Compares baseline vs optimized

**Output:** `data/results/benchmark_results.json`

## Generate Figures

### Create Publication-Quality Plots

```bash
python figures/generate_figures.py
```

**Creates:**
- Figure 1: Energy improvement (82.31%)
- Figure 2: Cache miss reduction (100%)
- Figure 3: Batch homogeneity (2,077%)
- Figure 4: Viability Index (714M)
- Figure 5: Statistical significance (p-values)

**Output:** `figures/figure_*.png` (DPI 300)

## Using as a Library

### Import and Use Components

```python
from code.parte_2_hypothesis_validation import SemanticCoherenceIndex, CacheMissEmulator
from code.parte_3_semantic_batching import TransformerWorkloadSimulator
from code.parte_4_overhead_analysis import OverheadAnalysis

# Compute semantic coherence
index = SemanticCoherenceIndex()
sigma = index.compute(operation_labels)  # Returns 0-1

# Simulate cache behavior
emulator = CacheMissEmulator()
cost, miss_rate = emulator.compute_cache_cost(operations)

# Simulate Transformer
simulator = TransformerWorkloadSimulator(embedding_dim=768, num_heads=12)
result = simulator.process_batch(batch)

# Analyze overhead
analysis = OverheadAnalysis()
results = analysis.measure_overhead()
print(f"VI = {results['viability']['viability_index']:.1e}")
```

## Reproduce Exact Results

### With Fixed Seeds

```bash
# Reproduces exact Part 2 results
python -c "
from code.parte_2_hypothesis_validation import Part2Experiment
exp = Part2Experiment(num_operations=100000, num_iterations=30, seed=42)
results = exp.run()
print(f'Energy improvement: {results[\"energy\"][\"improvement_percent\"]:.2f}%')
"
```

### Validate Reproducibility

```bash
bash scripts/reproduce_results.sh

# Output should show:
# ✓ All results match within numerical precision
# ✓ Reproducibility validation passed
```

## Batch Processing

### Run Multiple Scales

```bash
for scale in 10000 100000 1000000; do
  echo "Scale: $scale"
  python code/parte_2_hypothesis_validation.py --operations $scale
done
```

### Run with Custom Parameters

```bash
python code/parte_2_hypothesis_validation.py \
  --operations 50000 \
  --iterations 20 \
  --seed 123 \
  --output custom_results.json
```

## Jupyter Notebooks

### Interactive Analysis

```bash
jupyter notebook notebooks/01_Part_2_Analysis.ipynb
jupyter notebook notebooks/02_Part_3_Analysis.ipynb
jupyter notebook notebooks/03_Part_4_Analysis.ipynb
jupyter notebook notebooks/04_Complete_Results.ipynb
```

## Generate Report

```bash
python scripts/generate_report.py --output report.md

# Generates markdown report with:
# - Summary of results
# - Key metrics and improvements
# - Statistical validation
# - Visualizations
```

## Validate Environment

```bash
python scripts/validate_environment.py

# Checks:
# ✓ Python version
# ✓ NumPy, SciPy, matplotlib installed
# ✓ Disk space available
# ✓ GPU simulation capabilities
```

## Command-Line Interface

```bash
# Run benchmark
semantic-gpu-benchmark

# Validate environment
semantic-gpu-validate
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=code --cov-report=html

# Run specific test
pytest tests/test_semantic_coherence.py::test_perfect_homogeneity -v

# Run tests in parallel
pytest tests/ -n auto
```

## Troubleshooting

### Memory Error
```bash
# Reduce scale
python code/parte_2_hypothesis_validation.py --operations 10000
```

### Slow Execution
```bash
# Reduce iterations
python code/parte_2_hypothesis_validation.py --iterations 5
```

### Missing Output Files
```bash
# Create directories
mkdir -p data/results figures
python scripts/setup_environment.sh
```

## Next Steps

- Read [METHODOLOGY.md](METHODOLOGY.md) for technical details
- Check [RESULTS.md](RESULTS.md) for expected outputs
- Review [FAQ.md](FAQ.md) for common questions
