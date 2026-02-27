"""
Tests for reproducibility and consistency
"""

import numpy as np
import pytest
from code.parte_2_hypothesis_validation import Part2Experiment
from code.parte_3_semantic_batching import Part3Experiment


def test_part2_reproducibility_seed42():
    """Test that Part 2 produces same results with seed=42"""
    
    exp1 = Part2Experiment(num_operations=1000, num_iterations=3, seed=42)
    results1 = exp1.run()
    
    exp2 = Part2Experiment(num_operations=1000, num_iterations=3, seed=42)
    results2 = exp2.run()
    
    # Check energy improvement matches
    assert np.isclose(
        results1["energy"]["improvement_percent"],
        results2["energy"]["improvement_percent"],
        rtol=1e-5
    )
    
    # Check p-value matches
    assert np.isclose(
        results1["statistics"]["energy_t_test"]["p_value"],
        results2["statistics"]["energy_t_test"]["p_value"],
        rtol=1e-10
    )


def test_part3_reproducibility_seed42():
    """Test that Part 3 produces same results with seed=42"""
    
    exp1 = Part3Experiment(num_tokens=1000, num_iterations=2, seed=42)
    results1 = exp1.run()
    
    exp2 = Part3Experiment(num_tokens=1000, num_iterations=2, seed=42)
    results2 = exp2.run()
    
    # Check homogeneity means match
    assert np.isclose(
        results1["homogeneity"]["random_batching"]["mean"],
        results2["homogeneity"]["random_batching"]["mean"],
        rtol=1e-5
    )


def test_different_seeds_produce_different_results():
    """Test that different seeds produce different (but valid) results"""
    
    exp1 = Part2Experiment(num_operations=1000, num_iterations=3, seed=42)
    results1 = exp1.run()
    
    exp2 = Part2Experiment(num_operations=1000, num_iterations=3, seed=123)
    results2 = exp2.run()
    
    # Results should be different with different seeds
    assert not np.isclose(
        results1["energy"]["random_grouping_gj"],
        results2["energy"]["random_grouping_gj"],
        rtol=0.1
    )
    
    # But improvement should be in same ballpark (within 10%)
    assert np.isclose(
        results1["energy"]["improvement_percent"],
        results2["energy"]["improvement_percent"],
        rtol=0.1
    )


def test_statistical_significance_consistent():
    """Test that statistical significance is robust"""
    
    for seed in [42, 123, 456]:
        exp = Part2Experiment(num_operations=10000, num_iterations=10, seed=seed)
        results = exp.run()
        
        # Should always be significant (p < 0.05)
        assert results["statistics"]["energy_t_test"]["p_value"] < 0.05
        
        # Should always show substantial improvement (>50%)
        assert results["energy"]["improvement_percent"] > 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
