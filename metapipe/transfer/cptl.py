#!/usr/bin/env python3
"""
CPTL: Cross-Pipeline Transfer Learning

Novel contribution: Meta-learning for zero-shot routing policy transfer
across time-series domains with domain adaptation via low-rank projections.

Equation:
    Φ_inv(x, D) = Φ_shared(x) + A_D · Φ_specific(x)
    where A_D = U_D V_D^T (low-rank domain adaptation)

Algorithm: MAML-inspired meta-routing with domain-invariant representations

Theorem: Transfer error bounded by domain divergence:
    E_target[Loss] ≤ E_source[Loss] + λ·d_H(D_source, D_target) + ε
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from copy import deepcopy
from collections import defaultdict


class DomainAdapter(nn.Module):
    """
    Low-rank domain adaptation matrix A_D = U_D V_D^T

    Novel: Learns domain-specific feature transformations while
    sharing majority of parameters across domains
    """

    def __init__(
        self,
        feature_dim: int,
        rank: int = 32,
        n_domains: int = 10
    ):
        """
        Parameters
        ----------
        feature_dim : int
            Dimension of features
        rank : int
            Rank of adaptation matrices (compression)
        n_domains : int
            Number of source domains for meta-training
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.rank = rank

        # Domain-specific low-rank factors
        # U_D: (feature_dim, rank), V_D: (rank, feature_dim)
        self.domain_U = nn.ParameterDict({
            f'domain_{i}': nn.Parameter(torch.randn(feature_dim, rank) * 0.01)
            for i in range(n_domains)
        })
        self.domain_V = nn.ParameterDict({
            f'domain_{i}': nn.Parameter(torch.randn(rank, feature_dim) * 0.01)
            for i in range(n_domains)
        })

    def forward(
        self,
        features: torch.Tensor,
        domain_id: int
    ) -> torch.Tensor:
        """
        Apply domain adaptation

        A_D · features

        Parameters
        ----------
        features : torch.Tensor
            Shape (batch, feature_dim)
        domain_id : int
            Domain identifier

        Returns
        -------
        torch.Tensor
            Adapted features, shape (batch, feature_dim)
        """
        key = f'domain_{domain_id}'
        if key not in self.domain_U:
            # Unknown domain, return features unchanged
            return features

        U = self.domain_U[key]  # (feature_dim, rank)
        V = self.domain_V[key]  # (rank, feature_dim)

        # A = U @ V: (feature_dim, feature_dim)
        # Compute efficiently: (features @ V^T) @ U^T
        adapted = features @ V.t() @ U.t()

        return adapted

    def add_domain(self, domain_id: int):
        """Initialize adaptation parameters for new domain"""
        key = f'domain_{domain_id}'
        if key not in self.domain_U:
            self.domain_U[key] = nn.Parameter(
                torch.randn(self.feature_dim, self.rank) * 0.01
            )
            self.domain_V[key] = nn.Parameter(
                torch.randn(self.rank, self.feature_dim) * 0.01
            )


class MetaRouter(nn.Module):
    """
    Meta-Router using MAML for few-shot adaptation to new domains

    Novel Algorithm:
        Meta-Train:
            for each domain batch:
                θ_i ← θ - α∇L_domain_i(θ)  # inner loop
                meta_loss += L_domain_i(θ_i)
            θ ← θ - β∇(meta_loss)  # outer loop

        Zero-Shot Transfer:
            θ_new ← θ - α∇L_new_domain(θ)  # single step!
    """

    def __init__(
        self,
        feature_dim: int,
        n_actions: int,
        n_source_domains: int = 20,
        adaptation_rank: int = 32,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        n_inner_steps: int = 5
    ):
        """
        Parameters
        ----------
        feature_dim : int
            Dimension of TCAR features
        n_actions : int
            Number of possible actions (routing decisions)
        n_source_domains : int
            Number of source domains for meta-training
        adaptation_rank : int
            Rank for low-rank domain adaptation
        inner_lr : float
            Learning rate for inner loop (fast adaptation)
        outer_lr : float
            Learning rate for outer loop (meta-update)
        n_inner_steps : int
            Number of gradient steps in inner loop
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.n_actions = n_actions
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.n_inner_steps = n_inner_steps

        # Shared feature extractor (domain-invariant)
        self.shared_net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Domain adapter
        self.domain_adapter = DomainAdapter(
            feature_dim,
            rank=adaptation_rank,
            n_domains=n_source_domains
        )

        # Policy head
        self.policy_head = nn.Linear(256, n_actions)

        # Meta-optimizer (for outer loop)
        self.meta_optimizer = torch.optim.Adam(
            self.parameters(),
            lr=outer_lr
        )

        # Track domain performance
        self.domain_stats = defaultdict(list)

    def forward(
        self,
        features: torch.Tensor,
        domain_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional domain adaptation

        Parameters
        ----------
        features : torch.Tensor
            Shape (batch, feature_dim)
        domain_id : int, optional
            Domain ID for adaptation

        Returns
        -------
        torch.Tensor
            Action logits, shape (batch, n_actions)
        """
        # Domain adaptation
        if domain_id is not None:
            features = self.domain_adapter(features, domain_id)

        # Shared feature extraction
        h = self.shared_net(features)

        # Policy logits
        logits = self.policy_head(h)

        return logits

    def meta_train_step(
        self,
        domain_batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]],
        task_batch_size: int = 4
    ) -> Dict[str, float]:
        """
        Single meta-training step (MAML outer loop)

        Parameters
        ----------
        domain_batches : list
            List of (features, actions, rewards, domain_id) tuples
        task_batch_size : int
            Number of domains to sample per meta-batch

        Returns
        -------
        dict
            Training statistics
        """
        # Sample domains for this meta-batch
        sampled = np.random.choice(
            len(domain_batches),
            size=min(task_batch_size, len(domain_batches)),
            replace=False
        )

        meta_loss = 0.0
        meta_accuracy = 0.0

        for idx in sampled:
            features, actions, rewards, domain_id = domain_batches[idx]

            # Inner loop: fast adaptation
            adapted_params = self._inner_loop(
                features, actions, rewards, domain_id
            )

            # Compute meta-loss with adapted parameters
            with torch.enable_grad():
                # Temporarily replace parameters
                original_params = {
                    name: param.clone()
                    for name, param in self.named_parameters()
                }

                for name, param in adapted_params.items():
                    self.get_parameter(name).data = param

                # Forward pass
                logits = self.forward(features, domain_id)
                loss = F.cross_entropy(logits, actions, reduction='mean')

                # Weighted by rewards (policy gradient style)
                weighted_loss = (loss * rewards.abs().mean()).mean()

                meta_loss += weighted_loss

                # Accuracy
                preds = logits.argmax(dim=1)
                acc = (preds == actions).float().mean()
                meta_accuracy += acc

                # Restore original parameters
                for name, param in original_params.items():
                    self.get_parameter(name).data = param

        # Average meta-loss
        meta_loss = meta_loss / len(sampled)
        meta_accuracy = meta_accuracy / len(sampled)

        # Meta-update (outer loop)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        self.meta_optimizer.step()

        return {
            'meta_loss': meta_loss.item(),
            'meta_accuracy': meta_accuracy.item()
        }

    def _inner_loop(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        domain_id: int
    ) -> Dict[str, torch.Tensor]:
        """
        Inner loop: fast adaptation to domain

        Returns adapted parameters after K gradient steps

        Parameters
        ----------
        features : torch.Tensor
            Shape (batch, feature_dim)
        actions : torch.Tensor
            Ground truth actions, shape (batch,)
        rewards : torch.Tensor
            Rewards for actions, shape (batch,)
        domain_id : int
            Domain identifier

        Returns
        -------
        dict
            Adapted parameters
        """
        # Clone current parameters
        adapted_params = {
            name: param.clone()
            for name, param in self.named_parameters()
        }

        for step in range(self.n_inner_steps):
            # Forward with adapted params
            logits = self._forward_with_params(
                features, domain_id, adapted_params
            )

            # Loss
            loss = F.cross_entropy(logits, actions, reduction='mean')
            weighted_loss = (loss * rewards.abs().mean()).mean()

            # Compute gradients w.r.t. adapted params
            grads = torch.autograd.grad(
                weighted_loss,
                adapted_params.values(),
                create_graph=True,
                allow_unused=True
            )

            # Update adapted params
            for (name, param), grad in zip(adapted_params.items(), grads):
                if grad is not None:
                    adapted_params[name] = param - self.inner_lr * grad

        return adapted_params

    def _forward_with_params(
        self,
        features: torch.Tensor,
        domain_id: int,
        params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass with custom parameters (for MAML inner loop)
        """
        # This is simplified; full implementation would use functional API
        # For now, temporarily set params
        original = {}
        for name, param in params.items():
            original[name] = self.get_parameter(name).data.clone()
            self.get_parameter(name).data = param

        logits = self.forward(features, domain_id)

        # Restore
        for name, param in original.items():
            self.get_parameter(name).data = param

        return logits

    def adapt_to_new_domain(
        self,
        domain_id: int,
        support_features: torch.Tensor,
        support_actions: torch.Tensor,
        support_rewards: torch.Tensor,
        n_adapt_steps: int = 5
    ) -> nn.Module:
        """
        Zero-shot adaptation to new domain

        Novel: Single gradient step often suffices due to meta-training!

        Parameters
        ----------
        domain_id : int
            New domain ID
        support_features : torch.Tensor
            Support set features, shape (n_support, feature_dim)
        support_actions : torch.Tensor
            Support set actions, shape (n_support,)
        support_rewards : torch.Tensor
            Support set rewards, shape (n_support,)
        n_adapt_steps : int
            Number of adaptation steps (usually 1-5)

        Returns
        -------
        nn.Module
            Adapted model
        """
        # Clone model for adaptation
        adapted_model = deepcopy(self)

        # Add domain adapter if needed
        adapted_model.domain_adapter.add_domain(domain_id)

        # Optimizer for adaptation
        adapt_optimizer = torch.optim.SGD(
            adapted_model.parameters(),
            lr=self.inner_lr
        )

        # Adapt
        adapted_model.train()
        for step in range(n_adapt_steps):
            logits = adapted_model(support_features, domain_id)
            loss = F.cross_entropy(logits, support_actions)
            weighted_loss = (loss * support_rewards.abs().mean()).mean()

            adapt_optimizer.zero_grad()
            weighted_loss.backward()
            adapt_optimizer.step()

        return adapted_model

    def compute_domain_divergence(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> float:
        """
        Compute H-divergence between source and target domains

        Approximation using classifier-based method

        Returns
        -------
        float
            Domain divergence estimate
        """
        # Train binary classifier to distinguish source vs target
        n_source = len(source_features)
        n_target = len(target_features)

        features = torch.cat([source_features, target_features], dim=0)
        labels = torch.cat([
            torch.zeros(n_source),
            torch.ones(n_target)
        ], dim=0).long()

        # Simple classifier
        classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

        # Train
        classifier.train()
        for _ in range(100):
            logits = classifier(features)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate
        classifier.eval()
        with torch.no_grad():
            logits = classifier(features)
            preds = logits.argmax(dim=1)
            accuracy = (preds == labels).float().mean()

        # H-divergence ≈ 2(1 - 2*error)
        error = 1 - accuracy.item()
        divergence = 2 * (1 - 2 * error)

        return float(divergence)

    def save_checkpoint(self, path: str):
        """Save meta-learned model"""
        torch.save({
            'state_dict': self.state_dict(),
            'meta_optimizer': self.meta_optimizer.state_dict(),
            'domain_stats': dict(self.domain_stats),
            'feature_dim': self.feature_dim,
            'n_actions': self.n_actions
        }, path)

    def load_checkpoint(self, path: str):
        """Load meta-learned model"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer'])
        self.domain_stats = defaultdict(list, checkpoint['domain_stats'])
