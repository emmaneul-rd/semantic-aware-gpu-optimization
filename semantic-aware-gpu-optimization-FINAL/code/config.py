"""
Global configuration for Semantic-Aware GPU Optimization Framework
"""

# Random seed for reproducibility
RANDOM_SEED = 42

# Experiment configuration
PART_2_NUM_OPERATIONS = 100000
PART_2_NUM_ITERATIONS = 30
PART_2_NUM_CATEGORIES = 10

PART_3_NUM_TOKENS = 100000
PART_3_NUM_ITERATIONS = 10
PART_3_EMBEDDING_DIM = 768
PART_3_NUM_HEADS = 12
PART_3_BATCH_SIZE = 32

# Output paths
RESULTS_DIR = "data/results"
FIGURES_DIR = "figures"
NOTEBOOKS_DIR = "notebooks"
LOGS_DIR = "logs"

# Statistical parameters
SIGNIFICANCE_LEVEL = 0.05
MIN_SAMPLE_SIZE = 30

# GPU specs (H100)
GPU_SPECS = {
    "compute_capacity_tflops": 1456,
    "memory_bandwidth_tbs": 3.35,
    "l1_cache_kb": 256,
    "l2_cache_mb": 12,
    "hbm_bandwidth_gbs": 50,
    "thermal_limit_w": 700,
}

# Cache costs (in cycles)
CACHE_COSTS = {
    "L1": 1,
    "L2": 4,
    "L3": 12,
    "RAM": 40,
}
