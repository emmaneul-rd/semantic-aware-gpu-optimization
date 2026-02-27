# System Architecture

## Overview

The Semantic-Aware Execution framework consists of three main components:

```
┌─────────────────────────────────────────────────────────────┐
│                    Semantic Awareness Layer                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Semantic     │  │ Cache-Miss   │  │ Viability    │      │
│  │ Coherence    │  │ Emulator     │  │ Analysis     │      │
│  │ Index (σ)   │  │              │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Experimental Validation                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Part 2:      │  │ Part 3:      │  │ Part 4:      │      │
│  │ Operation-   │  │ Transformer- │  │ Overhead &   │      │
│  │ Level        │  │ Scale        │  │ Viability    │      │
│  │ (100K ops)   │  │ (100K tokens)│  │ (Theoretical)│      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      Synthesis & Publication                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Visualiza-   │  │ Reports &    │  │ Academic     │      │
│  │ tions        │  │ Analysis     │  │ Paper        │      │
│  │ (Figures)    │  │ (Markdown)   │  │ (IEEE/ACM)   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Semantic Coherence Index (σ)

**Location:** `code/parte_*.py`

**Purpose:** Quantifies batch homogeneity using Shannon entropy

**Formula:**
```
σ(B) = 1 - H(S) / log₂(|S|)
```

**Properties:**
- σ = 1.0: Perfect homogeneity
- σ = 0.0: Maximum entropy
- 0 ≤ σ ≤ 1: Always bounded

**Implementation:**
```python
def compute_sigma(labels):
    unique = len(set(labels))
    counts = {l: labels.count(l) for l in set(labels)}
    probs = [c/len(labels) for c in counts.values()]
    entropy = -sum(p * log2(p) for p in probs)
    return 1 - (entropy / log2(unique))
```

### 2. Cache-Miss Emulator

**Location:** `code/parte_2_hypothesis_validation.py`

**Purpose:** Models GPU memory hierarchy and cache behavior

**Memory Levels:**
```
L1 (256 KB/SM): 1 cycle
L2 (12 MB):     4 cycles  
L3 (50 MB):     12 cycles
HBM (80+ GB):   40 cycles
```

**Algorithm:**
1. For each operation, track semantic type
2. Check if same type accessed recently (within N operations)
3. If yes → data in L1/L2 → cost = 1-4 cycles
4. If no → cache miss → cost = 40 cycles

### 3. Viability Analysis

**Location:** `code/parte_4_overhead_analysis.py`

**Purpose:** Proves overhead is negligible

**Components:**
- Embedding generation: O(n × d)
- Clustering: O(n)
- Sorting: O(n log n)
- Coherence calculation: O(m)

**Viability Index:**
```
VI = Benefit / Overhead
VI = 1.96e16 / 27.5e6 = 713,881,904.6
```

## Experimental Pipeline

### Part 2: Operation-Level (100K Ops)

```python
Part2Experiment()
├── generate_operations()      # Synthetic ops with semantic labels
├── run_random_grouping()      # Baseline: no semantic info
├── run_semantic_grouping()    # Optimized: group by semantic type
├── statistical_analysis()     # t-test, Cohen's d, p-values
└── save_results()             # JSON output
```

**Output:** `data/results/part_2_results.json`

### Part 3: Transformer-Scale (100K Tokens)

```python
Part3Experiment()
├── generate_tokens()          # Realistic 768-dim embeddings
├── run_random_batching()      # Baseline: arbitrary batching
├── run_semantic_batching()    # Optimized: group by semantic label
├── statistical_analysis()     # Compute σ improvement
└── save_results()             # JSON output
```

**Output:** `data/results/part_3_results.json`

### Part 4: Overhead Analysis

```python
OverheadAnalysis()
├── measure_overhead()         # Classify, cluster, sort costs
├── compute_benefit()          # From Part 2 results
├── viability_index()          # Benefit / Overhead
├── complexity_analysis()      # O(n log n) proof
└── save_results()             # JSON output
```

**Output:** `data/results/part_4_results.json`

## Data Flow

```
Input Data (Synthetic)
        ↓
[Semantic Labeling]
        ↓
Labeled Operations
        ↓
[Part 2: Operation-Level]  →  Energy Data
        ↓
[Part 3: Transformer-Scale] → Homogeneity Data
        ↓
[Part 4: Overhead]          → Cost/Benefit Data
        ↓
[Statistical Analysis]
        ↓
Results (JSON)
        ↓
[Visualization & Reporting]
        ↓
Figures & Reports
        ↓
[Academic Paper]
```

## Key Algorithms

### Semantic Coherence Index

```python
def semantic_coherence(labels):
    """Compute σ = 1 - H(S)/log₂(k)"""
    unique_labels = set(labels)
    counts = {l: sum(1 for x in labels if x == l) for l in unique_labels}
    probs = np.array([c/len(labels) for c in counts.values()])
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = np.log2(len(unique_labels))
    return 1 - (entropy / max_entropy) if max_entropy > 0 else 1.0
```

### Cache Cost Computation

```python
def cache_cost(operations):
    """Compute memory access cost"""
    total_cost = 0
    for i, op in enumerate(operations):
        # Find recent same-type operations
        recent_same = sum(
            1 for j in range(max(0, i-10), i)
            if operations[j].type == op.type
        )
        
        # If recent → in cache, else → miss
        cost = L1_COST if recent_same > 0 else RAM_COST
        total_cost += cost * op.memory_footprint
    
    return total_cost
```

### Viability Index

```python
def viability_index(benefit_flops, overhead_flops):
    """VI = Benefit / Overhead"""
    return benefit_flops / max(overhead_flops, 1e-10)
```

## Testing Strategy

### Unit Tests

```python
tests/
├── test_semantic_coherence.py     # σ computation
├── test_cache_emulator.py         # Cache cost model
├── test_overhead_calculation.py   # Viability analysis
├── test_reproducibility.py        # Seed consistency
└── conftest.py                    # Pytest fixtures
```

### Integration Tests

- Full pipeline execution
- Results consistency across runs
- Statistical validation of results

### CI/CD

```yaml
.github/workflows/
├── tests.yml              # Unit tests + linting
├── reproducibility.yml    # Full pipeline validation
└── docs.yml              # Documentation build
```

## Configuration

### Default Parameters

```python
code/config.py:

RANDOM_SEED = 42
PART_2_NUM_OPERATIONS = 100000
PART_2_NUM_ITERATIONS = 30
PART_3_NUM_TOKENS = 100000
PART_3_NUM_ITERATIONS = 10
```

### Cache Model

```python
CACHE_COSTS = {
    "L1": 1,       # cycles
    "L2": 4,
    "L3": 12,
    "RAM": 40,
}
```

## Scalability

**Time Complexity:**
- Overall: O(n log n)
- Dominated by: Intra-cluster sorting
- Scalable to: 100M+ operations

**Space Complexity:**
- O(n) for embeddings
- O(k) for clusters (k << n)
- Memory efficient

## Extensibility

### Adding New Experiments

1. Subclass `Experiment` base class
2. Implement `run()` method
3. Save results to JSON
4. Add to pipeline

### Adding New Metrics

1. Define metric function
2. Compute in experiment
3. Add to results output
4. Visualize in figures

### Adding New Visualizations

1. Add figure generation function to `generate_figures.py`
2. Use matplotlib for consistency
3. Export DPI 300 PNG
4. Add caption to figure

## Design Principles

1. **Reproducibility First:** Fixed seeds, deterministic algorithms
2. **Clear Separation:** Core logic, experiments, visualization
3. **Open Source:** Complete code, no closed components
4. **Documentation:** Every function documented
5. **Testing:** Comprehensive test coverage
6. **Scalability:** O(n log n) algorithms
7. **Generalization:** Vendor-agnostic approach

## Future Extensions (Phase 2-4)

- CUDA/GPU implementation
- Real hardware validation
- Integration with TensorRT/vLLM
- ISA extension proposals
- Multi-GPU support
