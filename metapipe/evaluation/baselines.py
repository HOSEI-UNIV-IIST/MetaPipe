#!/usr/bin/env python3
"""
Baseline Routing Strategies for Comparison

Baselines:
1. Static Best: Oracle selects best single model (upper bound)
2. Random: Uniform random model selection
3. Greedy Quality: Always strongest/most accurate model
4. Greedy Cost: Always cheapest model
5. Thompson Sampling: Standard contextual bandit
6. Round-Robin: Cycle through models
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Any
from abc import ABC, abstractmethod


class BaseRouter(ABC):
    """Abstract base class for routing strategies"""

    @abstractmethod
    def select_model(self, features: np.ndarray, models: List[str]) -> int:
        """Select model index given features and available models"""
        pass

    @abstractmethod
    def update(self, features: np.ndarray, action: int, reward: float):
        """Update policy (if learnable)"""
        pass


class StaticBest(BaseRouter):
    """
    Oracle baseline: always selects the best single model

    Computed offline using validation set
    """

    def __init__(self, best_model_idx: int = 0):
        self.best_model_idx = best_model_idx

    def select_model(self, features: np.ndarray, models: List[str]) -> int:
        return self.best_model_idx

    def update(self, features: np.ndarray, action: int, reward: float):
        pass  # static policy


class RandomRouter(BaseRouter):
    """Uniform random model selection"""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def select_model(self, features: np.ndarray, models: List[str]) -> int:
        return self.rng.randint(0, len(models))

    def update(self, features: np.ndarray, action: int, reward: float):
        pass


class GreedyQuality(BaseRouter):
    """Always select the strongest/most accurate model"""

    def __init__(self, quality_ranking: List[int] = None):
        """
        Parameters
        ----------
        quality_ranking : list
            Indices of models sorted by quality (best first)
        """
        self.quality_ranking = quality_ranking or [0]

    def select_model(self, features: np.ndarray, models: List[str]) -> int:
        return self.quality_ranking[0]

    def update(self, features: np.ndarray, action: int, reward: float):
        pass


class GreedyCost(BaseRouter):
    """Always select the cheapest model"""

    def __init__(self, cost_ranking: List[int] = None):
        """
        Parameters
        ----------
        cost_ranking : list
            Indices of models sorted by cost (cheapest first)
        """
        self.cost_ranking = cost_ranking or [0]

    def select_model(self, features: np.ndarray, models: List[str]) -> int:
        return self.cost_ranking[0]

    def update(self, features: np.ndarray, action: int, reward: float):
        pass


class RoundRobin(BaseRouter):
    """Cycle through models in round-robin fashion"""

    def __init__(self):
        self.counter = 0

    def select_model(self, features: np.ndarray, models: List[str]) -> int:
        idx = self.counter % len(models)
        self.counter += 1
        return idx

    def update(self, features: np.ndarray, action: int, reward: float):
        pass


class ThompsonSampling(BaseRouter):
    """
    Thompson Sampling contextual bandit

    Uses Bayesian linear regression for reward modeling
    """

    def __init__(self, n_actions: int, feature_dim: int, alpha: float = 1.0):
        self.n_actions = n_actions
        self.feature_dim = feature_dim
        self.alpha = alpha  # prior variance

        # Posterior parameters per action
        self.A = [np.eye(feature_dim) for _ in range(n_actions)]  # precision
        self.b = [np.zeros(feature_dim) for _ in range(n_actions)]  # mean

    def select_model(self, features: np.ndarray, models: List[str]) -> int:
        """Sample from posterior and select argmax"""
        sampled_rewards = []

        for a in range(self.n_actions):
            # Posterior mean and covariance
            A_inv = np.linalg.inv(self.A[a])
            mu = A_inv @ self.b[a]

            # Sample from posterior
            sample = np.random.multivariate_normal(mu, self.alpha * A_inv)
            reward_estimate = features @ sample

            sampled_rewards.append(reward_estimate)

        return int(np.argmax(sampled_rewards))

    def update(self, features: np.ndarray, action: int, reward: float):
        """Bayesian update"""
        self.A[action] += np.outer(features, features)
        self.b[action] += reward * features


class EpsilonGreedy(BaseRouter):
    """
    Epsilon-greedy Q-learning

    Simple but effective baseline
    """

    def __init__(self, n_actions: int, feature_dim: int, epsilon: float = 0.1, lr: float = 0.01):
        self.n_actions = n_actions
        self.feature_dim = feature_dim
        self.epsilon = epsilon
        self.lr = lr

        # Linear Q-function per action
        self.weights = [np.zeros(feature_dim) for _ in range(n_actions)]

    def select_model(self, features: np.ndarray, models: List[str]) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_actions)

        # Greedy: argmax Q(s, a)
        q_values = [features @ w for w in self.weights]
        return int(np.argmax(q_values))

    def update(self, features: np.ndarray, action: int, reward: float):
        """Q-learning update"""
        q_current = features @ self.weights[action]
        td_error = reward - q_current
        self.weights[action] += self.lr * td_error * features
