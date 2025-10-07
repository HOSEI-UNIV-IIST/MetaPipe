#!/usr/bin/env python3
"""
Pytest configuration and fixtures for MetaPipe tests
"""

import pytest
import numpy as np
import torch


@pytest.fixture(scope='session', autouse=True)
def set_random_seeds():
    """Set random seeds for reproducibility"""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def sample_time_series():
    """Generate a sample time series for testing"""
    np.random.seed(42)
    t = np.linspace(0, 10, 200)
    series = (
        0.5 * t +  # trend
        10 * np.sin(2 * np.pi * t / 5) +  # seasonality
        np.random.randn(200) * 0.5  # noise
    )
    return series


@pytest.fixture
def sample_metadata():
    """Sample metadata for time series"""
    return {
        'domain': 'finance',
        'task_type': 'forecast',
        'horizon': 12,
        'frequency': 'daily',
        'seasonality': 7
    }
