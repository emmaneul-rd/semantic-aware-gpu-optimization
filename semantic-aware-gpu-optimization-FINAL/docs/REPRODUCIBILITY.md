# Reproducibility Guide

This guide explains how to reproduce the exact results from the paper.

## Quick Reproduction (5 minutes)

```bash
bash scripts/reproduce_results.sh
```

This runs all experiments with the exact same configuration as the paper.

## Detailed Reproduction

### Environment

Exact Python and dependencies:

```bash
python --version  # 3.8 or higher
pip install numpy>=1.21.0 scipy>=1.7.0 matplotlib>=3.4.0
```

### Part 2: Hypothesis Validation

```bash
python code/parte_2_hypothesis_validation.py
```

**Expected Output:**
- Energy improvement: 82.31% (±0.05%)
- p-value: 2.34e-154
- Cache miss reduction: 0.51% → 0.00%
- Cohen's d: 856.87

**Output file:** `data/results/part_2_results.json`

### Part 3: Transformer Batching

```bash
python code/parte_3_semantic_batching.py
```

**Expected Output:**
- Random σ: ~0.0459
- Semantic σ: ~1.0000
- Improvement: ~2,077%
- p-value: 1.01e-64

**Output file:** `data/results/part_3_results.json`

### Part 4: Overhead Analysis

```bash
python code/parte_4_overhead_analysis.py
```

**Expected Output:**
- Total overhead: 27.5M FLOPs
- Viability Index: 713,881,904.6
- Overhead as % of benefit: 0.00000014%

**Output file:** `data/results/part_4_results.json`

## Numerical Tolerances

Results should match with:
- Relative tolerance: 1e-5 (0.001%)
- Absolute tolerance: 1e-8

```python
import numpy as np
np.allclose(expected, actual, rtol=1e-5, atol=1e-8)
```

## Random Seed

All experiments use `seed=42`:

```python
np.random.seed(42)
```

This ensures exact reproducibility. Different seeds will produce slightly different 
results but with similar magnitudes.

## Verification Script

```bash
python scripts/validate_results.py
```

This automatically verifies that your results match the expected values.

## Hardware Notes

- **CPU:** Tested on Intel Core i7/i9
- **GPU:** Not required (CPU simulation)
- **Memory:** >2 GB recommended
- **Disk:** ~1 GB for code + results

## Troubleshooting

### "Different results than paper"

Check:
1. Python version: `python --version` (must be 3.8+)
2. NumPy version: `python -c "import numpy; print(numpy.__version__)"` (must be 1.21+)
3. Random seed: Ensure you're using `seed=42`
4. Numerical precision: Use tolerance levels above

### "Out of memory"

Reduce the scale:

```python
# Default: 100,000 operations
# Reduced: 10,000 operations
python code/parte_2_hypothesis_validation.py --operations 10000
```

### "Missing dependencies"

```bash
pip install -r code/requirements.txt
```

## Publication Figures

Generate publication-quality figures:

```bash
python figures/generate_figures.py --dpi 300 --format png
```

Expected figures:
- figure_1_energy_improvement.png (82.31% improvement)
- figure_2_cache_miss_reduction.png (100% reduction)
- figure_3_batch_homogeneity.png (2,077% improvement)
- figure_4_viability_index.png (VI = 714M)
- figure_5_statistical_significance.png (p-values)

## Code Reproducibility

All source code is version-controlled and reproducible:

```bash
# Check code version
git log --oneline -1

# Verify code hasn't changed
git diff

# Clone fresh copy
git clone https://github.com/emmaneul-rd/semantic-aware-gpu-optimization.git
```

## Data Reproducibility

Synthetic data is generated deterministically:

```python
from code.data.synthetic.generate_data import generate_operations

# Same seed → same data
ops = generate_operations(100000, seed=42)
```

## Statistical Validation

All results include statistical measures:

```json
{
  "energy_t_test": {
    "t_statistic": 3262.86,
    "p_value": 2.34e-154,
    "significant": true
  },
  "effect_size_cohens_d": 856.87
}
```

Verify:
- t-statistic > 2.0 (significant difference)
- p-value < 0.05 (reject null hypothesis)
- Cohen's d > 0.8 (large effect size)

## Long-Term Reproducibility

For archival purposes:

```bash
# Generate checksums
sha256sum code/*.py > code_checksums.txt

# Verify later
sha256sum -c code_checksums.txt
```

## Variations That Don't Affect Reproducibility

These should produce nearly identical results:

- Different CPU processor (different speed, same semantics)
- Different operating system (Windows, macOS, Linux)
- Different installation paths (code is relative-path agnostic)

These WILL affect results:

- Different random seed (controlled by --seed)
- Different Python version (must be 3.8+)
- Different NumPy/SciPy versions (must match requirements.txt)
- Different experiment parameters (--operations, --iterations)

## Citation for Reproducible Research

If you use this framework:

```bibtex
@article{sanchez2026semantic,
  title={Semantic-Aware Execution: Memory-Optimal GPU Computing Through Data-Centric Optimization},
  author={Sánchez Pache, Emmanuel},
  journal={IEEE Transactions on Computers},
  year={2026},
  note={Code and reproducibility guide: https://github.com/emmaneul-rd/semantic-aware-gpu-optimization}
}
```

## Questions?

- Check [FAQ.md](FAQ.md)
- Open an [Issue](https://github.com/emmaneul-rd/semantic-aware-gpu-optimization/issues)
- Contact: emmanuel@salomoncoral.com
