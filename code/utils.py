"""
Utility functions for Semantic-Aware GPU Optimization Framework
"""

import numpy as np
from typing import List, Dict, Tuple


def shannon_entropy(probabilities: np.ndarray) -> float:
    """
    Calculate Shannon entropy of a probability distribution.
    
    Args:
        probabilities: Array of probabilities
        
    Returns:
        Shannon entropy value
    """
    probs = np.asarray(probabilities)
    probs = probs[probs > 0]  # Remove zero probabilities
    return -np.sum(probs * np.log2(probs + 1e-10))


def compute_label_distribution(labels: List[str]) -> Dict[str, float]:
    """
    Compute probability distribution of labels.
    
    Args:
        labels: List of semantic labels
        
    Returns:
        Dictionary mapping labels to probabilities
    """
    unique_labels = np.unique(labels)
    distribution = {}
    
    for label in unique_labels:
        count = sum(1 for l in labels if l == label)
        distribution[label] = count / len(labels)
    
    return distribution


def create_output_directories() -> None:
    """Create necessary output directories if they don't exist."""
    import os
    from .config import RESULTS_DIR, FIGURES_DIR, LOGS_DIR, NOTEBOOKS_DIR
    
    for directory in [RESULTS_DIR, FIGURES_DIR, LOGS_DIR, NOTEBOOKS_DIR]:
        os.makedirs(directory, exist_ok=True)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    L2 normalize embeddings.
    
    Args:
        embeddings: Array of shape (n, d)
        
    Returns:
        Normalized embeddings
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-10)


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix.
    
    Args:
        embeddings: Array of shape (n, d)
        
    Returns:
        Similarity matrix of shape (n, n)
    """
    normalized = normalize_embeddings(embeddings)
    return np.dot(normalized, normalized.T)


def group_by_label(items: List, labels: List[str]) -> Dict[str, List]:
    """
    Group items by their labels.
    
    Args:
        items: List of items to group
        labels: List of labels (same length as items)
        
    Returns:
        Dictionary mapping labels to lists of items
    """
    groups = {}
    for item, label in zip(items, labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(item)
    
    return groups


def flatten_list_of_lists(lol: List[List]) -> List:
    """Flatten list of lists to single list."""
    return [item for sublist in lol for item in sublist]


def compute_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Compute confidence interval for mean.
    
    Args:
        data: Array of values
        confidence: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    from scipy import stats
    
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    ci = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    
    return mean - ci, mean + ci


def save_results_json(data: Dict, filepath: str) -> None:
    """
    Save results to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Output filepath
    """
    import json
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_results_json(filepath: str) -> Dict:
    """
    Load results from JSON file.
    
    Args:
        filepath: Input filepath
        
    Returns:
        Dictionary from JSON
    """
    import json
    
    with open(filepath, 'r') as f:
        return json.load(f)


def print_results_table(results: Dict, title: str = "Results") -> None:
    """
    Pretty-print results dictionary as table.
    
    Args:
        results: Dictionary of results
        title: Table title
    """
    print("\n" + "="*70)
    print(f"{title:^70}")
    print("="*70)
    
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.6f}")
                else:
                    print(f"  {k}: {v}")
        else:
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")
    
    print("="*70 + "\n")
