#!/usr/bin/env python3
"""
Tests for MAP (Multi-Horizon Adaptive Policy)
"""

import pytest
import numpy as np
import torch
from metapipe.policy.map import MAPPolicy, QNetwork, HorizonAggregator, HorizonReward


class TestQNetwork:
    """Test Q-Network"""

    def test_forward_pass(self):
        """Test forward pass through Q-network"""
        feature_dim = 50
        n_actions = 5
        batch_size = 32

        net = QNetwork(feature_dim, n_actions)
        features = torch.randn(batch_size, feature_dim)

        q_values = net(features)

        assert q_values.shape == (batch_size, n_actions)

    def test_output_values_finite(self):
        """Test that Q-values are finite"""
        net = QNetwork(50, 5)
        features = torch.randn(10, 50)
        q_values = net(features)

        assert torch.all(torch.isfinite(q_values))


class TestHorizonAggregator:
    """Test Horizon Aggregator"""

    def test_fixed_weighting(self):
        """Test fixed exponential decay weighting"""
        n_horizons = 5
        feature_dim = 50

        aggregator = HorizonAggregator(n_horizons, feature_dim, mode='fixed')

        # Check weights sum to 1
        assert torch.isclose(aggregator.fixed_weights.sum(), torch.tensor(1.0))

    def test_adaptive_weighting(self):
        """Test adaptive learned weighting"""
        n_horizons = 5
        feature_dim = 50
        batch_size = 32
        n_actions = 5

        aggregator = HorizonAggregator(n_horizons, feature_dim, mode='adaptive')

        # Dummy Q-values
        q_per_horizon = [torch.randn(batch_size, n_actions) for _ in range(n_horizons)]
        features = torch.randn(batch_size, feature_dim)

        aggregated = aggregator(q_per_horizon, features)

        assert aggregated.shape == (batch_size, n_actions)

    def test_aggregation_output_finite(self):
        """Test aggregated outputs are finite"""
        aggregator = HorizonAggregator(5, 50, mode='fixed')
        q_per_horizon = [torch.randn(10, 5) for _ in range(5)]

        aggregated = aggregator(q_per_horizon)

        assert torch.all(torch.isfinite(aggregated))


class TestMAPPolicy:
    """Test MAP Policy"""

    @pytest.fixture
    def policy(self):
        """Create MAP policy instance"""
        return MAPPolicy(
            feature_dim=50,
            n_actions=5,
            horizons=[1, 3, 6, 12, 24],
            gamma=0.99,
            learning_rate=1e-3
        )

    def test_initialization(self, policy):
        """Test policy initialization"""
        assert len(policy.horizon_heads) == policy.n_horizons
        assert policy.n_actions == 5
        assert len(policy.horizons) == 5

    def test_select_action(self, policy):
        """Test action selection"""
        features = np.random.randn(50)
        action = policy.select_action(features, epsilon=0.0)

        assert 0 <= action < policy.n_actions
        assert isinstance(action, int)

    def test_epsilon_greedy_exploration(self, policy):
        """Test epsilon-greedy exploration"""
        features = np.random.randn(50)

        # With epsilon=1.0, should always explore (random action)
        actions = [policy.select_action(features, epsilon=1.0) for _ in range(100)]
        unique_actions = len(set(actions))

        # Should have some variety
        assert unique_actions > 1

    def test_forward_aggregate(self, policy):
        """Test forward aggregation"""
        batch_size = 32
        features = torch.randn(batch_size, policy.feature_dim)

        q_aggregated = policy._forward_aggregate(features)

        assert q_aggregated.shape == (batch_size, policy.n_actions)

    def test_update_policy(self, policy):
        """Test policy update"""
        features = np.random.randn(50)
        action = 2
        rewards_per_horizon = [
            HorizonReward(horizon=h, reward=0.5, quality=0.8, cost=0.1, latency=100)
            for h in policy.horizons
        ]
        next_features = np.random.randn(50)

        # Should not crash
        policy.update(features, action, rewards_per_horizon, next_features, done=False)

    def test_replay_buffer(self, policy):
        """Test replay buffer functionality"""
        for i in range(10):
            features = np.random.randn(50)
            action = i % policy.n_actions
            rewards = [
                HorizonReward(horizon=h, reward=0.5, quality=0.8, cost=0.1, latency=100)
                for h in policy.horizons
            ]
            next_features = np.random.randn(50)

            policy.update(features, action, rewards, next_features, done=False)

        assert len(policy.replay_buffer) == 10

    def test_save_load(self, policy, tmp_path):
        """Test save and load functionality"""
        save_path = tmp_path / "policy.pt"

        # Save
        policy.save(str(save_path))
        assert save_path.exists()

        # Load
        new_policy = MAPPolicy(
            feature_dim=50,
            n_actions=5,
            horizons=[1, 3, 6, 12, 24]
        )
        new_policy.load(str(save_path))

        # Check parameters match
        features = torch.randn(1, 50)
        q1 = policy._forward_aggregate(features)
        q2 = new_policy._forward_aggregate(features)

        assert torch.allclose(q1, q2)


class TestMAPIntegration:
    """Integration tests for MAP"""

    def test_learning_improves_performance(self):
        """Test that learning improves Q-values over time"""
        policy = MAPPolicy(feature_dim=10, n_actions=3, horizons=[1, 3, 6])

        # Fixed state and action
        state = np.random.randn(10)
        action = 0

        # Initial Q-value
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_initial = policy._forward_aggregate(state_t)[0, action].item()

        # Train with positive rewards
        for _ in range(100):
            rewards = [
                HorizonReward(horizon=h, reward=1.0, quality=0.9, cost=0.1, latency=100)
                for h in policy.horizons
            ]
            next_state = np.random.randn(10)
            policy.update(state, action, rewards, next_state, done=False)

        # Final Q-value should increase
        with torch.no_grad():
            q_final = policy._forward_aggregate(state_t)[0, action].item()

        # Q-value should have changed (learning occurred)
        assert abs(q_final - q_initial) > 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
