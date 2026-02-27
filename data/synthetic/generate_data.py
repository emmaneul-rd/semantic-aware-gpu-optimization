#!/usr/bin/env python3
"""
Synthetic data generation for reproducible experiments
"""

import numpy as np
import json
from pathlib import Path


def generate_operations(
    num_operations: int = 100000,
    num_categories: int = 10,
    seed: int = 42,
    output_dir: str = "data/synthetic"
) -> dict:
    """
    Generate synthetic GPU operations with semantic labels.
    
    Args:
        num_operations: Number of operations to generate
        num_categories: Number of semantic categories
        seed: Random seed for reproducibility
        output_dir: Directory to save data
        
    Returns:
        Dictionary with generated data
    """
    
    np.random.seed(seed)
    
    # Semantic categories
    categories = [f"category_{i}" for i in range(num_categories)]
    
    # Generate operations
    operations = {
        "op_ids": list(range(num_operations)),
        "semantic_labels": [
            np.random.choice(categories) for _ in range(num_operations)
        ],
        "memory_footprints": np.random.randint(
            100, 10000, num_operations
        ).tolist(),
        "config": {
            "num_operations": num_operations,
            "num_categories": num_categories,
            "seed": seed,
        }
    }
    
    # Save to disk
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / f"operations_{num_operations}.json"
    
    with open(output_file, 'w') as f:
        json.dump(operations, f)
    
    print(f"✓ Generated {num_operations} operations → {output_file}")
    
    return operations


def generate_tokens(
    num_tokens: int = 100000,
    embedding_dim: int = 768,
    num_categories: int = 20,
    seed: int = 42,
    output_dir: str = "data/synthetic"
) -> dict:
    """
    Generate synthetic Transformer tokens.
    
    Args:
        num_tokens: Number of tokens
        embedding_dim: Embedding dimension (typically 768)
        num_categories: Number of semantic categories
        seed: Random seed
        output_dir: Directory to save data
        
    Returns:
        Dictionary with generated tokens
    """
    
    np.random.seed(seed)
    
    # Categories
    categories = [f"token_type_{i}" for i in range(num_categories)]
    
    # Generate tokens
    tokens = {
        "token_ids": list(range(num_tokens)),
        "semantic_labels": [
            np.random.choice(categories) for _ in range(num_tokens)
        ],
        "embeddings_shape": [num_tokens, embedding_dim],
        "config": {
            "num_tokens": num_tokens,
            "embedding_dim": embedding_dim,
            "num_categories": num_categories,
            "seed": seed,
        }
    }
    
    # Save metadata (not full embeddings for size)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir) / f"tokens_{num_tokens}.json"
    
    with open(output_file, 'w') as f:
        json.dump(tokens, f)
    
    print(f"✓ Generated {num_tokens} tokens → {output_file}")
    
    return tokens


def verify_data_consistency(operations: dict, num_samples: int = 100):
    """
    Verify generated data has expected properties.
    
    Args:
        operations: Dictionary of operations
        num_samples: Number of samples to check
    """
    
    # Check counts match
    assert len(operations["op_ids"]) == operations["config"]["num_operations"]
    assert len(operations["semantic_labels"]) == operations["config"]["num_operations"]
    assert len(operations["memory_footprints"]) == operations["config"]["num_operations"]
    
    # Check values are in expected ranges
    footprints = operations["memory_footprints"]
    assert min(footprints) >= 100
    assert max(footprints) < 10000
    
    # Check categories
    unique_categories = set(operations["semantic_labels"])
    assert len(unique_categories) <= operations["config"]["num_categories"]
    
    print(f"✓ Data consistency verified")
    print(f"  - Total operations: {len(operations['op_ids']):,}")
    print(f"  - Unique categories: {len(unique_categories)}")
    print(f"  - Memory footprint range: [{min(footprints)}, {max(footprints)}]")


def main():
    """Generate all synthetic datasets"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--operations", type=int, default=100000, help="Num operations")
    parser.add_argument("--tokens", type=int, default=100000, help="Num tokens")
    parser.add_argument("--output", default="data/synthetic", help="Output directory")
    
    args = parser.parse_args()
    
    print("Generating synthetic data for reproducible experiments...")
    print("="*70)
    
    # Generate operations
    ops = generate_operations(
        num_operations=args.operations,
        num_categories=10,
        seed=args.seed,
        output_dir=args.output
    )
    
    # Verify
    verify_data_consistency(ops)
    
    # Generate tokens
    toks = generate_tokens(
        num_tokens=args.tokens,
        embedding_dim=768,
        num_categories=20,
        seed=args.seed,
        output_dir=args.output
    )
    
    print("\n✅ Synthetic data generation complete")
    print(f"   Operations saved: {args.output}/operations_{args.operations}.json")
    print(f"   Tokens saved: {args.output}/tokens_{args.tokens}.json")


if __name__ == "__main__":
    main()
