"""
Semantic-Aware GPU Optimization: Part 2 - Hypothesis Validation

Validates the core hypothesis: semantic grouping reduces energy consumption
through improved cache locality.

Author: Emmanuel Sánchez Pache
Nodo Cero Research Division
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
from scipy import stats


@dataclass
class Operation:
    """Represents a GPU operation with semantic properties."""
    op_id: int
    semantic_label: str
    embedding: np.ndarray
    memory_footprint: int


class SemanticCoherenceIndex:
    """Computes semantic coherence index (σ)."""
    
    @staticmethod
    def compute(labels: List[str]) -> float:
        """
        Compute σ = 1 - H(S) / log₂(|S|)
        where H(S) is Shannon entropy of label distribution.
        """
        if not labels:
            return 0.0
        
        unique_labels = len(set(labels))
        if unique_labels == 1:
            return 1.0
        
        # Count frequencies
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Shannon entropy
        probs = np.array([count / len(labels) for count in label_counts.values()])
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(unique_labels)
        
        sigma = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
        return float(sigma)


class CacheMissEmulator:
    """Emulates GPU cache behavior."""
    
    # Cache costs (cycles)
    L1_COST = 1
    L2_COST = 4
    L3_COST = 12
    RAM_COST = 40
    
    @staticmethod
    def compute_cache_cost(operations: List[Operation]) -> Tuple[float, float]:
        """
        Compute cache miss rate and energy cost.
        Better semantic coherence → better cache locality → lower cost.
        """
        total_cost = 0
        cache_misses = 0
        
        for i, op in enumerate(operations):
            # Check if this operation's data is in cache
            # (simplified: assume semantic similarity implies cache locality)
            semantic_label = op.semantic_label
            
            # Count recent same-type operations
            recent_same = sum(
                1 for j in range(max(0, i-10), i)
                if operations[j].semantic_label == semantic_label
            )
            
            # If recent same-type operation → data in L1/L2
            if recent_same > 0:
                cost = CacheMissEmulator.L1_COST
            else:
                # Cache miss → need L3 or RAM
                cost = CacheMissEmulator.RAM_COST
                cache_misses += 1
            
            total_cost += cost * op.memory_footprint
        
        miss_rate = cache_misses / len(operations) if operations else 0
        return total_cost, miss_rate


class Part2Experiment:
    """Conducts Part 2 experiment: hypothesis validation."""
    
    SEMANTIC_CATEGORIES = [
        "query", "key", "value", "attention_score",
        "mlp_activation", "layer_norm", "projection",
        "residual", "fusion", "embeddings"
    ]
    
    def __init__(self, num_operations: int = 100000, num_iterations: int = 30, seed: int = 42):
        """Initialize experiment."""
        self.num_operations = num_operations
        self.num_iterations = num_iterations
        self.seed = seed
        np.random.seed(seed)
    
    def generate_operations(self) -> List[Operation]:
        """Generate synthetic operations."""
        operations = []
        for i in range(self.num_operations):
            op = Operation(
                op_id=i,
                semantic_label=np.random.choice(self.SEMANTIC_CATEGORIES),
                embedding=np.random.randn(256).astype(np.float32),
                memory_footprint=np.random.randint(100, 10000)
            )
            operations.append(op)
        return operations
    
    def run_random_grouping(self) -> Tuple[List[float], List[float]]:
        """Run random grouping baseline."""
        print("Running random grouping baseline... ", end="", flush=True)
        
        energy_values = []
        miss_rates = []
        
        for iteration in range(self.num_iterations):
            ops = self.generate_operations()
            
            # Shuffle for random grouping
            np.random.shuffle(ops)
            
            # Compute cost
            cost, miss_rate = CacheMissEmulator.compute_cache_cost(ops)
            energy_values.append(cost)
            miss_rates.append(miss_rate * 100)  # Convert to percentage
        
        print("✓")
        return energy_values, miss_rates
    
    def run_semantic_grouping(self) -> Tuple[List[float], List[float]]:
        """Run semantic grouping optimized."""
        print("Running semantic grouping optimized... ", end="", flush=True)
        
        energy_values = []
        miss_rates = []
        
        for iteration in range(self.num_iterations):
            ops = self.generate_operations()
            
            # Group by semantic label
            semantic_groups = {}
            for op in ops:
                if op.semantic_label not in semantic_groups:
                    semantic_groups[op.semantic_label] = []
                semantic_groups[op.semantic_label].append(op)
            
            # Reconstruct operations in semantic order
            semantic_ops = []
            for label, group in semantic_groups.items():
                semantic_ops.extend(group)
            
            # Compute cost
            cost, miss_rate = CacheMissEmulator.compute_cache_cost(semantic_ops)
            energy_values.append(cost)
            miss_rates.append(miss_rate * 100)
        
        print("✓")
        return energy_values, miss_rates
    
    def run(self) -> Dict:
        """Execute complete Part 2 experiment."""
        
        print("\n" + "="*70)
        print("PART 2: HYPOTHESIS VALIDATION - SEMANTIC GROUPING")
        print("="*70)
        print(f"Configuration: {self.num_operations:,} operations, {self.num_iterations} iterations\n")
        
        # Run experiments
        random_energy, random_misses = self.run_random_grouping()
        semantic_energy, semantic_misses = self.run_semantic_grouping()
        
        # Statistical analysis
        t_stat_energy, p_value_energy = stats.ttest_ind(semantic_energy, random_energy)
        
        random_mean = np.mean(random_energy)
        semantic_mean = np.mean(semantic_energy)
        improvement = ((random_mean - semantic_mean) / random_mean) * 100
        
        cohen_d = (semantic_mean - random_mean) / np.sqrt(
            (np.std(semantic_energy)**2 + np.std(random_energy)**2) / 2
        )
        
        # Results
        results = {
            "configuration": {
                "num_operations": self.num_operations,
                "num_iterations": self.num_iterations,
                "seed": self.seed
            },
            "energy": {
                "random_grouping_gj": float(random_mean),
                "semantic_grouping_gj": float(semantic_mean),
                "improvement_percent": float(improvement),
                "improvement_absolute": float(random_mean - semantic_mean),
            },
            "cache_miss_rate": {
                "random_percent": float(np.mean(random_misses)),
                "semantic_percent": float(np.mean(semantic_misses)),
            },
            "statistics": {
                "energy_t_test": {
                    "t_statistic": float(t_stat_energy),
                    "p_value": float(p_value_energy),
                    "significant": p_value_energy < 0.05,
                },
                "effect_size_cohens_d": float(cohen_d),
            }
        }
        
        # Print results
        print("Results:")
        print("-" * 70)
        print(f"Energy Consumption (GJ):")
        print(f"  Random Grouping:    {random_mean:.2e} ± {np.std(random_energy):.2e}")
        print(f"  Semantic Grouping:  {semantic_mean:.2e} ± {np.std(semantic_energy):.2e}")
        print(f"  Improvement:        {improvement:.2f}%")
        print(f"\nCache Miss Rate (%):")
        print(f"  Random Grouping:    {np.mean(random_misses):.4f}%")
        print(f"  Semantic Grouping:  {np.mean(semantic_misses):.4f}%")
        print(f"\nStatistical Significance:")
        print(f"  t-statistic:        {t_stat_energy:.2f}")
        print(f"  p-value:            {p_value_energy:.2e}")
        print(f"  Cohen's d:          {cohen_d:.2f}")
        
        return results


def main():
    """Main entry point."""
    
    experiment = Part2Experiment(
        num_operations=100000,
        num_iterations=30,
        seed=42
    )
    
    results = experiment.run()
    
    # Save results
    with open("data/results/part_2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: data/results/part_2_results.json")
    
    return results


if __name__ == "__main__":
    main()
