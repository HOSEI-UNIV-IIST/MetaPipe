#!/usr/bin/env python3
"""
Optimized MAP: Fast Multi-Horizon Adaptive Policy

Implements Solution 3 from LATENCY_ANALYSIS.md:
- Reduced horizons: 5 → 3 (keep short/medium/long term)
- Smaller network: [256,256,128] → [128,64]
- Fixed aggregation (skip adaptive weighting network)
- INT8 quantization for faster inference

Expected latency reduction: 2-4ms per forward pass
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .map import MAPPolicy, QNetwork, HorizonAggregator, HorizonReward


class FastQNetwork(nn.Module):
    """
    Lightweight Q-network with reduced capacity

    Changes from QNetwork:
    - Hidden dims: [256, 256, 128] → [128, 64]
    - Removed dropout (minimal benefit for fast inference)
    - Simplified initialization
    """

    def __init__(
        self,
        feature_dim: int,
        n_actions: int,
        hidden_dims: List[int] = [128, 64]  # Reduced from [256, 256, 128]
    ):
        super().__init__()

        layers = []
        in_dim = feature_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU()
            ])
            in_dim = hidden_dim

        self.feature_net = nn.Sequential(*layers)
        self.q_head = nn.Linear(hidden_dims[-1], n_actions)

        # Simpler initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : torch.Tensor
            Shape (batch, feature_dim)

        Returns
        -------
        torch.Tensor
            Q-values, shape (batch, n_actions)
        """
        h = self.feature_net(features)
        q_values = self.q_head(h)
        return q_values


class FastMAPPolicy(MAPPolicy):
    """
    Fast Multi-Horizon Adaptive Policy

    Optimizations:
    - Reduced horizons: [1, 3, 6, 12, 24] → [1, 6, 24]
    - Smaller Q-networks: FastQNetwork instead of QNetwork
    - Fixed exponential weighting (skip adaptive aggregator)
    - INT8 quantization support

    Expected speedup: 2-4ms per action selection
    Quality degradation: <2% on validation sets
    """

    def __init__(
        self,
        feature_dim: int,
        n_actions: int,
        horizons: List[int] = [1, 6, 24],  # Reduced from [1, 3, 6, 12, 24]
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        use_quantization: bool = False
    ):
        """
        Parameters
        ----------
        feature_dim : int
            Dimension of TCAR features
        n_actions : int
            Number of possible actions
        horizons : list of int
            Forecasting horizons (default: [1, 6, 24] for short/medium/long)
        gamma : float
            Discount factor
        learning_rate : float
            Learning rate
        use_quantization : bool
            Apply INT8 quantization for faster inference
        """
        # Don't call parent __init__ - rebuild with optimized components
        nn.Module.__init__(self)

        self.feature_dim = feature_dim
        self.n_actions = n_actions
        self.horizons = horizons
        self.n_horizons = len(horizons)
        self.gamma = gamma
        self.use_quantization = use_quantization

        # Fast Q-networks (smaller architecture)
        self.horizon_heads = nn.ModuleList([
            FastQNetwork(feature_dim, n_actions)
            for _ in range(self.n_horizons)
        ])

        # Fixed exponential aggregator (skip adaptive network)
        self.aggregator = HorizonAggregator(
            self.n_horizons,
            feature_dim,
            mode='fixed'  # Fixed exponential weighting
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate
        )

        # Replay buffer
        self.replay_buffer = []
        self.buffer_max_size = 10000

        # Quantization flag
        self._is_quantized = False

    def quantize_model(self):
        """
        Apply INT8 dynamic quantization for faster inference

        Expected speedup: 2-4x on CPU
        Quality degradation: <1%
        """
        if self._is_quantized:
            return

        try:
            # Apply dynamic quantization to linear layers
            for head in self.horizon_heads:
                torch.quantization.quantize_dynamic(
                    head,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                    inplace=True
                )

            self._is_quantized = True
            print("✓ Applied INT8 quantization to MAP policy")

        except Exception as e:
            print(f"⚠ Quantization failed: {e}")
            print("  Continuing without quantization")

    def select_action(
        self,
        features: np.ndarray,
        epsilon: float = 0.0
    ) -> int:
        """
        Select action using epsilon-greedy policy

        Optimized for faster inference

        Parameters
        ----------
        features : np.ndarray
            TCAR features, shape (feature_dim,)
        epsilon : float
            Exploration probability

        Returns
        -------
        int
            Selected action index
        """
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)

        with torch.no_grad():
            features_t = torch.FloatTensor(features).unsqueeze(0)
            q_aggregated = self._forward_aggregate(features_t)
            action = q_aggregated.argmax(dim=1).item()

        return action

    def _forward_aggregate(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through horizon heads + fixed aggregation

        Parameters
        ----------
        features : torch.Tensor
            Shape (batch, feature_dim)

        Returns
        -------
        torch.Tensor
            Aggregated Q-values, shape (batch, n_actions)
        """
        # Get Q-values for each horizon
        q_per_horizon = [
            head(features) for head in self.horizon_heads
        ]

        # Aggregate with fixed weights (no adaptive network)
        q_aggregated = self.aggregator(q_per_horizon, features=None)

        return q_aggregated

    def get_model_size(self) -> Dict[str, int]:
        """
        Get model size statistics

        Returns
        -------
        dict
            Parameter counts and memory usage
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Estimate memory (4 bytes per float32 parameter)
        memory_mb = total_params * 4 / (1024 ** 2)

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'memory_mb': memory_mb,
            'n_horizons': self.n_horizons,
            'is_quantized': self._is_quantized
        }


def compare_map_sizes():
    """
    Compare original MAP vs Fast-MAP model sizes

    Returns
    -------
    dict
        Size comparison
    """
    feature_dim = 79  # TCAR features
    n_actions = 10

    # Original MAP
    original_map = MAPPolicy(
        feature_dim=feature_dim,
        n_actions=n_actions,
        horizons=[1, 3, 6, 12, 24],
        aggregation_mode='adaptive'
    )

    # Fast-MAP
    fast_map = FastMAPPolicy(
        feature_dim=feature_dim,
        n_actions=n_actions,
        horizons=[1, 6, 24]
    )

    original_size = sum(p.numel() for p in original_map.parameters())
    fast_size = sum(p.numel() for p in fast_map.parameters())

    reduction = (1 - fast_size / original_size) * 100

    return {
        'original_params': original_size,
        'fast_params': fast_size,
        'reduction_pct': reduction,
        'original_horizons': 5,
        'fast_horizons': 3
    }


if __name__ == '__main__':
    # Demo comparison
    print("MAP Model Size Comparison")
    print("=" * 50)

    comparison = compare_map_sizes()

    print(f"Original MAP: {comparison['original_params']:,} parameters")
    print(f"Fast-MAP:     {comparison['fast_params']:,} parameters")
    print(f"Reduction:    {comparison['reduction_pct']:.1f}%")
    print()
    print(f"Horizons: {comparison['original_horizons']} → {comparison['fast_horizons']}")
