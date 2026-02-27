"""
Semantic-Aware GPU Optimization Framework

A compiler-level paradigm for improving GPU memory efficiency through
semantic awareness of data access patterns.

Author: Emmanuel Sánchez Pache
Nodo Cero Research Division
"""

__version__ = "1.0.0"
__author__ = "Emmanuel Sánchez Pache"
__license__ = "Apache-2.0"

from . import parte_2_hypothesis_validation
from . import parte_3_semantic_batching
from . import parte_4_overhead_analysis
from . import benchmark_simulation

__all__ = [
    "parte_2_hypothesis_validation",
    "parte_3_semantic_batching",
    "parte_4_overhead_analysis",
    "benchmark_simulation",
]
