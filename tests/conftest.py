"""
Pytest configuration and shared fixtures
"""

import pytest
import numpy as np


@pytest.fixture
def random_seed():
    """Fixture providing random seed for reproducibility"""
    return 42


@pytest.fixture
def sample_operations():
    """Fixture providing sample operations for testing"""
    semantic_labels = ["query", "key", "value", "mlp", "norm"]
    return [
        {"label": np.random.choice(semantic_labels), "size": np.random.randint(100, 1000)}
        for _ in range(100)
    ]


@pytest.fixture
def sample_embeddings():
    """Fixture providing sample embeddings"""
    np.random.seed(42)
    return np.random.randn(10, 768).astype(np.float32)


@pytest.fixture(autouse=True)
def set_random_seed():
    """Automatically set random seed for all tests"""
    np.random.seed(42)
    yield
    # Cleanup after test


def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        if "slow" in item.nodeid:
            item.add_marker(pytest.mark.slow)
