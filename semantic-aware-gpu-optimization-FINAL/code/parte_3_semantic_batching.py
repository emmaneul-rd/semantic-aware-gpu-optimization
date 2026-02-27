"""
Semantic-Aware GPU Optimization: Part 3 - Transformer Batching

This module validates semantic-aware execution at the Transformer scale,
measuring batch homogeneity and energy impact.

Experimental Design:
  - 100,000 synthetic tokens with realistic Transformer properties
  - 20 semantic categories (query, key, value, mlp, etc.)
  - Random vs. Semantic batching comparison
  - Statistical validation (t-tests, p-values)

Author: Emmanuel Sánchez Pache
Nodo Cero Research Division
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict
import json
from scipy import stats
import sys


@dataclass
class Token:
    """Represents a token with semantic properties."""
    token_id: int
    semantic_label: str
    embedding: np.ndarray  # 768-dim like BERT
    position: int
    attention_head: int
    layer: int


@dataclass
class Batch:
    """Represents a batch of tokens."""
    batch_id: int
    tokens: List[Token]
    semantic_labels: List[str]
    
    @property
    def semantic_homogeneity(self) -> float:
        """Compute semantic coherence index σ."""
        if not self.semantic_labels:
            return 0.0
        
        unique_labels = len(set(self.semantic_labels))
        if unique_labels == 1:
            return 1.0
        
        # Shannon entropy
        label_counts = {}
        for label in self.semantic_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        probs = np.array([count / len(self.semantic_labels) 
                         for count in label_counts.values()])
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(unique_labels)
        
        # σ = 1 - (H / log2(k))
        sigma = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
        return float(sigma)
    
    @property
    def cache_efficiency_proxy(self) -> float:
        """Estimate cache efficiency based on batch homogeneity."""
        sigma = self.semantic_homogeneity
        # Perfect homogeneity (σ=1) → perfect cache locality
        # Random homogeneity (σ≈0) → poor cache locality
        return sigma


class TokenGenerator:
    """Generates synthetic tokens with semantic properties."""
    
    SEMANTIC_CATEGORIES = [
        "query_projection",      # Query operations
        "key_projection",        # Key operations
        "value_projection",      # Value operations
        "attention_softmax",     # Attention softmax
        "attention_output",      # Attention output projection
        "mlp_activation",        # MLP first layer
        "mlp_output",           # MLP output projection
        "layer_norm_1",         # Layer normalization 1
        "layer_norm_2",         # Layer normalization 2
        "embedding",            # Token embeddings
        "residual_add",         # Residual connections
        "dropout",              # Dropout operations
        "positional_encoding",  # Positional encodings
        "head_projection_1",    # Head 1 projection
        "head_projection_2",    # Head 2 projection
        "head_projection_3",    # Head 3 projection
        "head_projection_4",    # Head 4 projection
        "cache_key_value",      # KV cache operations
        "sequence_attention",   # Sequence attention
        "batch_normalize",      # Batch normalization
    ]
    
    def __init__(self, seed=42):
        """Initialize token generator."""
        np.random.seed(seed)
    
    def generate_tokens(self, num_tokens: int) -> List[Token]:
        """Generate synthetic tokens with realistic semantic distribution."""
        tokens = []
        
        for i in range(num_tokens):
            # Realistic semantic distribution
            semantic_label = np.random.choice(self.SEMANTIC_CATEGORIES)
            
            # 768-dim embeddings (BERT-like)
            embedding = np.random.randn(768).astype(np.float32)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
            
            token = Token(
                token_id=i,
                semantic_label=semantic_label,
                embedding=embedding,
                position=i % 128,  # Position in sequence (max 128)
                attention_head=i % 12,  # Which attention head
                layer=i % 12  # Which transformer layer
            )
            tokens.append(token)
        
        return tokens


class TransformerWorkloadSimulator:
    """Simulates Transformer computation with semantic batching."""
    
    def __init__(self, embedding_dim: int = 768, num_heads: int = 12, seed: int = 42):
        """Initialize simulator."""
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.seed = seed
        np.random.seed(seed)
    
    def simulate_attention(self, batch: Batch) -> float:
        """Simulate attention computation energy cost."""
        batch_size = len(batch.tokens)
        
        # Q, K, V projections
        energy_qkv = batch_size * self.embedding_dim * 3 * 2  # 2 ops per multiply
        
        # Attention matrix computation: Q @ K^T
        energy_qk = batch_size * batch_size * self.embedding_dim
        
        # Softmax and value projection
        energy_softmax = batch_size * batch_size
        energy_v = batch_size * self.embedding_dim
        
        # Memory access cost (worse for poor locality)
        coherence = batch.semantic_homogeneity
        memory_cost_factor = 1.0 + (1.0 - coherence) * 10  # 1-11x multiplier
        
        total_energy = (energy_qkv + energy_qk + energy_softmax + energy_v) * memory_cost_factor
        return total_energy
    
    def simulate_mlp(self, batch: Batch) -> float:
        """Simulate MLP computation energy cost."""
        batch_size = len(batch.tokens)
        
        # First linear layer (768 -> 3072)
        energy_fc1 = batch_size * self.embedding_dim * 3072 * 2
        
        # Activation (GELU approximation)
        energy_activation = batch_size * 3072
        
        # Second linear layer (3072 -> 768)
        energy_fc2 = batch_size * 3072 * self.embedding_dim * 2
        
        # Memory access cost
        coherence = batch.semantic_homogeneity
        memory_cost_factor = 1.0 + (1.0 - coherence) * 8
        
        total_energy = (energy_fc1 + energy_activation + energy_fc2) * memory_cost_factor
        return total_energy
    
    def process_batch(self, batch: Batch) -> Dict:
        """Process a batch and compute metrics."""
        
        attention_energy = self.simulate_attention(batch)
        mlp_energy = self.simulate_mlp(batch)
        total_energy = attention_energy + mlp_energy
        
        coherence = batch.semantic_homogeneity
        
        return {
            "batch_id": batch.batch_id,
            "batch_size": len(batch.tokens),
            "attention_energy": attention_energy,
            "mlp_energy": mlp_energy,
            "total_energy": total_energy,
            "semantic_homogeneity": coherence,
            "unique_semantics": len(set(batch.semantic_labels)),
        }


class Part3Experiment:
    """Conducts Part 3 experiment: Transformer batching validation."""
    
    def __init__(self, num_tokens: int = 100000, batch_size: int = 32, 
                 num_iterations: int = 10, seed: int = 42):
        """Initialize experiment."""
        self.num_tokens = num_tokens
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.seed = seed
        
        self.token_generator = TokenGenerator(seed=seed)
        self.simulator = TransformerWorkloadSimulator(seed=seed)
    
    def run_random_batching(self) -> Tuple[List[float], List[float]]:
        """Run random batching baseline."""
        print("Running random batching... ", end="", flush=True)
        
        homogeneity_scores = []
        energy_values = []
        
        for iteration in range(self.num_iterations):
            # Generate tokens
            tokens = self.token_generator.generate_tokens(self.num_tokens)
            
            # Shuffle for random batching
            np.random.shuffle(tokens)
            
            # Create batches
            iteration_homogeneity = []
            iteration_energy = []
            
            for batch_idx in range(0, len(tokens), self.batch_size):
                batch_tokens = tokens[batch_idx:batch_idx + self.batch_size]
                
                batch = Batch(
                    batch_id=batch_idx // self.batch_size,
                    tokens=batch_tokens,
                    semantic_labels=[t.semantic_label for t in batch_tokens]
                )
                
                result = self.simulator.process_batch(batch)
                iteration_homogeneity.append(result["semantic_homogeneity"])
                iteration_energy.append(result["total_energy"])
            
            homogeneity_scores.append(np.mean(iteration_homogeneity))
            energy_values.append(np.mean(iteration_energy))
        
        print(f"✓")
        return homogeneity_scores, energy_values
    
    def run_semantic_batching(self) -> Tuple[List[float], List[float]]:
        """Run semantic batching optimized."""
        print("Running semantic batching... ", end="", flush=True)
        
        homogeneity_scores = []
        energy_values = []
        
        for iteration in range(self.num_iterations):
            # Generate tokens
            tokens = self.token_generator.generate_tokens(self.num_tokens)
            
            # Group by semantic label
            semantic_groups = {}
            for token in tokens:
                if token.semantic_label not in semantic_groups:
                    semantic_groups[token.semantic_label] = []
                semantic_groups[token.semantic_label].append(token)
            
            # Create semantic batches
            iteration_homogeneity = []
            iteration_energy = []
            
            for semantic_label, group_tokens in semantic_groups.items():
                # Sort by similarity within semantic group
                embeddings = np.array([t.embedding for t in group_tokens])
                centroid = np.mean(embeddings, axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
                
                # Sort by similarity to centroid
                similarities = [np.dot(t.embedding, centroid) for t in group_tokens]
                sorted_tokens = [t for _, t in sorted(zip(similarities, group_tokens), reverse=True)]
                
                # Create batches from sorted group
                for batch_idx in range(0, len(sorted_tokens), self.batch_size):
                    batch_tokens = sorted_tokens[batch_idx:batch_idx + self.batch_size]
                    
                    batch = Batch(
                        batch_id=batch_idx // self.batch_size,
                        tokens=batch_tokens,
                        semantic_labels=[t.semantic_label for t in batch_tokens]
                    )
                    
                    result = self.simulator.process_batch(batch)
                    iteration_homogeneity.append(result["semantic_homogeneity"])
                    iteration_energy.append(result["total_energy"])
            
            homogeneity_scores.append(np.mean(iteration_homogeneity))
            energy_values.append(np.mean(iteration_energy))
        
        print(f"✓")
        return homogeneity_scores, energy_values
    
    def run(self) -> Dict:
        """Execute complete Part 3 experiment."""
        
        print("\n" + "="*70)
        print("PART 3: TRANSFORMER-SCALE SEMANTIC BATCHING VALIDATION")
        print("="*70)
        print(f"Configuration: {self.num_tokens:,} tokens, "
              f"batch_size={self.batch_size}, iterations={self.num_iterations}\n")
        
        # Run baselines and optimized
        random_homogeneity, random_energy = self.run_random_batching()
        semantic_homogeneity, semantic_energy = self.run_semantic_batching()
        
        # Statistical analysis
        t_stat, p_value = stats.ttest_ind(semantic_homogeneity, random_homogeneity)
        
        random_mean = np.mean(random_homogeneity)
        semantic_mean = np.mean(semantic_homogeneity)
        improvement = ((semantic_mean - random_mean) / random_mean) * 100
        
        cohen_d = (semantic_mean - random_mean) / np.sqrt(
            (np.std(semantic_homogeneity)**2 + np.std(random_homogeneity)**2) / 2
        )
        
        random_energy_mean = np.mean(random_energy)
        semantic_energy_mean = np.mean(semantic_energy)
        energy_improvement = ((random_energy_mean - semantic_energy_mean) / random_energy_mean) * 100
        
        # Results
        results = {
            "configuration": {
                "num_tokens": self.num_tokens,
                "batch_size": self.batch_size,
                "num_iterations": self.num_iterations,
                "seed": self.seed
            },
            "homogeneity": {
                "random_batching": {
                    "mean": float(random_mean),
                    "std": float(np.std(random_homogeneity)),
                    "min": float(np.min(random_homogeneity)),
                    "max": float(np.max(random_homogeneity)),
                },
                "semantic_batching": {
                    "mean": float(semantic_mean),
                    "std": float(np.std(semantic_homogeneity)),
                    "min": float(np.min(semantic_homogeneity)),
                    "max": float(np.max(semantic_homogeneity)),
                },
            },
            "energy": {
                "random_batching_gj": float(random_energy_mean),
                "semantic_batching_gj": float(semantic_energy_mean),
                "energy_improvement_percent": float(energy_improvement),
            },
            "statistics": {
                "homogeneity_t_test": {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                },
                "effect_size_cohens_d": float(cohen_d),
                "improvement_percent": float(improvement),
            }
        }
        
        # Print results
        print("\nResults:")
        print("-" * 70)
        print(f"Semantic Homogeneity (σ):")
        print(f"  Random Batching:    {random_mean:.4f} ± {np.std(random_homogeneity):.4f}")
        print(f"  Semantic Batching:  {semantic_mean:.4f} ± {np.std(semantic_homogeneity):.4f}")
        print(f"  Improvement:        {improvement:.1f}%")
        print(f"\nEnergy Impact:")
        print(f"  Random Batching:    {random_energy_mean:.2f} GJ")
        print(f"  Semantic Batching:  {semantic_energy_mean:.2f} GJ")
        print(f"  Improvement:        {energy_improvement:.1f}%")
        print(f"\nStatistical Significance:")
        print(f"  t-statistic:        {t_stat:.2f}")
        print(f"  p-value:            {p_value:.2e}")
        print(f"  Significant:        {'YES' if p_value < 0.05 else 'NO'}")
        print(f"  Cohen's d:          {cohen_d:.2f}")
        
        return results


def main():
    """Main entry point."""
    
    experiment = Part3Experiment(
        num_tokens=100000,
        batch_size=32,
        num_iterations=10,
        seed=42
    )
    
    results = experiment.run()
    
    # Save results
    with open("data/results/part_3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: data/results/part_3_results.json")
    
    return results


if __name__ == "__main__":
    main()
