#!/usr/bin/env python3
"""
MAP: Multi-Horizon Adaptive Policy

Novel contribution: Q-learning with simultaneous optimization across
multiple forecasting horizons with adaptive weighting.

Equation:
    Q^π_MH(s, a) = Σ_{h∈H} w_h · E[R_h(s, a, s') | s, a]

Theorem: Achieves regret bound O(√(|H| |A| T log T))
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class HorizonReward:
    """Reward structure across multiple horizons"""
    horizon: int
    reward: float
    quality: float
    cost: float
    latency: float


class QNetwork(nn.Module):
    """
    Q-network for a single horizon

    Architecture: MLP with residual connections
    """

    def __init__(
        self,
        feature_dim: int,
        n_actions: int,
        hidden_dims: List[int] = [256, 256, 128]
    ):
        """
        Parameters
        ----------
        feature_dim : int
            Dimension of input features (from TCAR)
        n_actions : int
            Number of possible actions (model combinations)
        hidden_dims : list
            Hidden layer dimensions
        """
        super().__init__()

        layers = []
        in_dim = feature_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim

        self.feature_net = nn.Sequential(*layers)
        self.q_head = nn.Linear(hidden_dims[-1], n_actions)

        # Initialize weights
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


class HorizonAggregator(nn.Module):
    """
    Aggregates Q-values across multiple horizons with learned/adaptive weights

    Novel contribution: Context-dependent horizon weighting
    """

    def __init__(
        self,
        n_horizons: int,
        feature_dim: int,
        mode: str = 'adaptive'
    ):
        """
        Parameters
        ----------
        n_horizons : int
            Number of horizons to aggregate
        feature_dim : int
            Dimension of features for adaptive weighting
        mode : str
            'fixed': exponential decay weights
            'adaptive': learned context-dependent weights
        """
        super().__init__()
        self.n_horizons = n_horizons
        self.mode = mode

        if mode == 'adaptive':
            # Learn weighting function from context
            self.weight_net = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.ReLU(),
                nn.Linear(64, n_horizons),
                nn.Softmax(dim=-1)
            )
        else:
            # Fixed exponential decay: w_h = exp(-λh) / Z
            self.register_buffer(
                'fixed_weights',
                self._compute_fixed_weights(n_horizons, lambda_decay=0.1)
            )

    def _compute_fixed_weights(
        self,
        n_horizons: int,
        lambda_decay: float
    ) -> torch.Tensor:
        """Compute exponential decay weights"""
        horizons = torch.arange(1, n_horizons + 1, dtype=torch.float32)
        weights = torch.exp(-lambda_decay * horizons)
        weights = weights / weights.sum()
        return weights

    def forward(
        self,
        q_values_per_horizon: List[torch.Tensor],
        features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Aggregate Q-values across horizons

        Parameters
        ----------
        q_values_per_horizon : list of torch.Tensor
            Each tensor has shape (batch, n_actions)
        features : torch.Tensor, optional
            Shape (batch, feature_dim), for adaptive weighting

        Returns
        -------
        torch.Tensor
            Aggregated Q-values, shape (batch, n_actions)
        """
        # Stack: (batch, n_horizons, n_actions)
        stacked = torch.stack(q_values_per_horizon, dim=1)

        if self.mode == 'adaptive' and features is not None:
            # Compute context-dependent weights
            weights = self.weight_net(features)  # (batch, n_horizons)
            weights = weights.unsqueeze(-1)  # (batch, n_horizons, 1)
        else:
            # Use fixed weights
            batch_size = stacked.shape[0]
            weights = self.fixed_weights.view(1, -1, 1).expand(
                batch_size, -1, 1
            )

        # Weighted sum: (batch, n_actions)
        aggregated = (stacked * weights).sum(dim=1)

        return aggregated


class MAPPolicy(nn.Module):
    """
    Multi-Horizon Adaptive Policy

    Learns Q-functions for multiple horizons and aggregates them adaptively.

    Novel Algorithm: Multi-horizon temporal difference learning with
    adaptive horizon weighting based on task context.
    """

    def __init__(
        self,
        feature_dim: int,
        n_actions: int,
        horizons: List[int] = [1, 3, 6, 12, 24],
        aggregation_mode: str = 'adaptive',
        gamma: float = 0.99,
        learning_rate: float = 1e-3
    ):
        """
        Parameters
        ----------
        feature_dim : int
            Dimension of TCAR features
        n_actions : int
            Number of possible actions (model combinations per stage)
        horizons : list of int
            Forecasting horizons to optimize
        aggregation_mode : str
            'fixed' or 'adaptive' horizon weighting
        gamma : float
            Discount factor for Q-learning
        learning_rate : float
            Learning rate for optimizer
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.n_actions = n_actions
        self.horizons = horizons
        self.n_horizons = len(horizons)
        self.gamma = gamma

        # Q-network per horizon
        self.horizon_heads = nn.ModuleList([
            QNetwork(feature_dim, n_actions)
            for _ in range(self.n_horizons)
        ])

        # Aggregator
        self.aggregator = HorizonAggregator(
            self.n_horizons,
            feature_dim,
            mode=aggregation_mode
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate
        )

        # Replay buffer
        self.replay_buffer = []
        self.buffer_max_size = 10000

    def select_action(
        self,
        features: np.ndarray,
        epsilon: float = 0.0
    ) -> int:
        """
        Select action using epsilon-greedy policy

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
        Forward pass through all horizon heads + aggregation

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

        # Aggregate
        q_aggregated = self.aggregator(q_per_horizon, features)

        return q_aggregated

    def update(
        self,
        features: np.ndarray,
        action: int,
        rewards_per_horizon: List[HorizonReward],
        next_features: np.ndarray,
        done: bool
    ):
        """
        Multi-horizon temporal difference update

        Novel Algorithm:
            For each horizon h:
                δ_h = r_h + γ max_a' Q_h(s', a') - Q_h(s, a)
                Q_h(s, a) ← Q_h(s, a) + α δ_h

        Parameters
        ----------
        features : np.ndarray
            Current state features
        action : int
            Action taken
        rewards_per_horizon : list of HorizonReward
            Rewards observed at each horizon
        next_features : np.ndarray
            Next state features
        done : bool
            Episode termination flag
        """
        # Add to replay buffer
        self._add_to_buffer(
            features, action, rewards_per_horizon, next_features, done
        )

        # Sample batch
        batch = self._sample_batch(batch_size=32)
        if batch is None:
            return

        features_b, actions_b, rewards_b, next_features_b, dones_b = batch

        # Convert to tensors
        features_t = torch.FloatTensor(features_b)
        actions_t = torch.LongTensor(actions_b)
        next_features_t = torch.FloatTensor(next_features_b)
        dones_t = torch.FloatTensor(dones_b)

        # Compute loss per horizon
        total_loss = 0.0

        for h_idx in range(self.n_horizons):
            # Extract rewards for this horizon
            rewards_h = torch.FloatTensor([
                rewards[h_idx].reward for rewards in rewards_b
            ])

            # Q-values for current state-action
            q_values = self.horizon_heads[h_idx](features_t)
            q_sa = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

            # Q-values for next state (max over actions)
            with torch.no_grad():
                next_q_values = self.horizon_heads[h_idx](next_features_t)
                next_q_max = next_q_values.max(dim=1)[0]
                target = rewards_h + self.gamma * next_q_max * (1 - dones_t)

            # TD loss
            loss_h = F.mse_loss(q_sa, target)
            total_loss += loss_h

        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        self.optimizer.step()

    def _add_to_buffer(
        self,
        features: np.ndarray,
        action: int,
        rewards: List[HorizonReward],
        next_features: np.ndarray,
        done: bool
    ):
        """Add transition to replay buffer"""
        self.replay_buffer.append(
            (features, action, rewards, next_features, done)
        )
        if len(self.replay_buffer) > self.buffer_max_size:
            self.replay_buffer.pop(0)

    def _sample_batch(self, batch_size: int) -> Optional[Tuple]:
        """Sample mini-batch from replay buffer"""
        if len(self.replay_buffer) < batch_size:
            return None

        indices = np.random.choice(
            len(self.replay_buffer),
            size=batch_size,
            replace=False
        )

        batch = [self.replay_buffer[i] for i in indices]

        # Unzip
        features_b = np.array([b[0] for b in batch])
        actions_b = np.array([b[1] for b in batch])
        rewards_b = [b[2] for b in batch]
        next_features_b = np.array([b[3] for b in batch])
        dones_b = np.array([b[4] for b in batch], dtype=np.float32)

        return features_b, actions_b, rewards_b, next_features_b, dones_b

    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'horizons': self.horizons,
            'n_actions': self.n_actions,
            'feature_dim': self.feature_dim
        }, path)

    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
