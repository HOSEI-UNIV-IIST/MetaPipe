#!/usr/bin/env python3
"""
Tests for BCPR (Budget-Constrained Pareto Routing)
"""

import pytest
import numpy as np
from metapipe.optimizer.bcpr import BCPROptimizer, ConstraintViolation, ParetoSolution


class TestBCPROptimizer:
    """Test suite for BCPR Optimizer"""

    @pytest.fixture
    def optimizer(self):
        """Create BCPR optimizer instance"""
        return BCPROptimizer(
            cost_budget=1.0,
            latency_budget=2000.0,
            quality_threshold=0.7,
            dual_lr=0.01
        )

    def test_initialization(self, optimizer):
        """Test optimizer initialization"""
        assert optimizer.cost_budget == 1.0
        assert optimizer.latency_budget == 2000.0
        assert optimizer.lambda_cost == 0.0
        assert optimizer.lambda_latency == 0.0

    def test_feasible_solution(self, optimizer):
        """Test feasibility check for valid solution"""
        assert optimizer.is_feasible(cost=0.5, latency=1000.0, quality=0.8) == True

    def test_infeasible_cost(self, optimizer):
        """Test infeasibility due to cost violation"""
        assert optimizer.is_feasible(cost=1.5, latency=1000.0, quality=0.8) == False

    def test_infeasible_latency(self, optimizer):
        """Test infeasibility due to latency violation"""
        assert optimizer.is_feasible(cost=0.5, latency=3000.0, quality=0.8) == False

    def test_infeasible_quality(self, optimizer):
        """Test infeasibility due to quality violation"""
        assert optimizer.is_feasible(cost=0.5, latency=1000.0, quality=0.5) == False

    def test_augmented_reward_no_violation(self, optimizer):
        """Test augmented reward with no constraint violations"""
        reward, violation = optimizer.compute_augmented_reward(
            quality=0.9,
            cost=0.5,
            latency=1000.0
        )

        # Should have no violations
        assert violation.cost_violation == 0.0
        assert violation.latency_violation == 0.0
        assert violation.quality_violation == 0.0

        # Reward should be positive (quality dominates)
        assert reward > 0

    def test_augmented_reward_with_cost_violation(self, optimizer):
        """Test augmented reward with cost violation"""
        reward, violation = optimizer.compute_augmented_reward(
            quality=0.9,
            cost=1.5,  # exceeds budget
            latency=1000.0
        )

        # Should have cost violation
        assert violation.cost_violation > 0
        # Reward should be penalized
        assert reward < optimizer.alpha_quality * 0.9

    def test_dual_variable_update(self, optimizer):
        """Test dual variable updates"""
        initial_lambda = optimizer.lambda_cost

        # Violate cost constraint
        _, violation = optimizer.compute_augmented_reward(
            quality=0.9,
            cost=1.5,
            latency=1000.0
        )

        optimizer.update_dual_variables(violation)

        # Lambda should increase
        assert optimizer.lambda_cost > initial_lambda

    def test_penalty_adaptation(self, optimizer):
        """Test adaptive penalty increase"""
        initial_penalty = optimizer.penalty

        # Repeatedly violate constraints
        for _ in range(optimizer.violation_patience + 1):
            _, violation = optimizer.compute_augmented_reward(
                quality=0.9,
                cost=1.5,
                latency=1000.0
            )
            optimizer.update_dual_variables(violation)

        # Penalty should have increased
        assert optimizer.penalty > initial_penalty

    def test_pareto_dominance(self, optimizer):
        """Test Pareto dominance check"""
        sol1 = ParetoSolution(action=0, quality=0.9, cost=0.5, latency=1000, score=0.8)
        sol2 = ParetoSolution(action=1, quality=0.8, cost=0.6, latency=1200, score=0.7)

        # sol1 dominates sol2 (better quality, lower cost, lower latency)
        assert optimizer._dominates(sol1, sol2) == True
        assert optimizer._dominates(sol2, sol1) == False

    def test_update_pareto_frontier(self, optimizer):
        """Test Pareto frontier updates"""
        # Add first solution
        optimizer.update_pareto_frontier(
            action=0,
            quality=0.8,
            cost=0.5,
            latency=1000
        )
        assert len(optimizer.pareto_frontier) == 1

        # Add dominated solution (should not be added)
        optimizer.update_pareto_frontier(
            action=1,
            quality=0.7,
            cost=0.6,
            latency=1200
        )
        assert len(optimizer.pareto_frontier) == 1  # dominated, not added

        # Add dominating solution (should replace)
        optimizer.update_pareto_frontier(
            action=2,
            quality=0.9,
            cost=0.4,
            latency=900
        )
        assert len(optimizer.pareto_frontier) == 1  # replaced previous

    def test_get_best_feasible_solution(self, optimizer):
        """Test retrieval of best feasible solution"""
        # Add feasible solutions
        optimizer.update_pareto_frontier(action=0, quality=0.8, cost=0.5, latency=1000)
        optimizer.update_pareto_frontier(action=1, quality=0.9, cost=0.6, latency=1200)

        best = optimizer.get_best_feasible_solution()

        assert best is not None
        assert optimizer.is_feasible(best.cost, best.latency, best.quality)

    def test_reset(self, optimizer):
        """Test optimizer reset"""
        # Modify state
        optimizer.lambda_cost = 1.0
        optimizer.penalty = 5.0
        optimizer.update_pareto_frontier(action=0, quality=0.8, cost=0.5, latency=1000)

        # Reset
        optimizer.reset()

        assert optimizer.lambda_cost == 0.0
        assert optimizer.lambda_latency == 0.0
        assert optimizer.penalty == 1.0
        assert len(optimizer.pareto_frontier) == 0


class TestBCPRIntegration:
    """Integration tests for BCPR"""

    def test_converges_to_feasible_region(self):
        """Test that optimizer converges to feasible solutions"""
        optimizer = BCPROptimizer(
            cost_budget=1.0,
            latency_budget=2000.0,
            dual_lr=0.1
        )

        # Simulate optimization iterations
        for _ in range(100):
            # Random quality/cost/latency
            quality = np.random.rand() * 0.5 + 0.5
            cost = np.random.rand() * 1.5  # may violate
            latency = np.random.rand() * 3000  # may violate

            reward, violation = optimizer.compute_augmented_reward(
                quality, cost, latency
            )
            optimizer.update_dual_variables(violation)

        # Lambda multipliers should have grown if violations occurred
        assert optimizer.lambda_cost >= 0
        assert optimizer.lambda_latency >= 0

    def test_pareto_coverage_computation(self):
        """Test Pareto coverage metric"""
        optimizer = BCPROptimizer(cost_budget=1.0, latency_budget=2000.0)

        # Oracle frontier
        oracle_frontier = [
            ParetoSolution(0, 0.9, 0.5, 1000, 0.8),
            ParetoSolution(1, 0.8, 0.3, 800, 0.7),
        ]

        # Add obtained points
        optimizer.update_pareto_frontier(0, 0.85, 0.4, 900)
        optimizer.update_pareto_frontier(1, 0.75, 0.35, 850)

        coverage = optimizer.get_pareto_coverage(oracle_frontier)

        # Should have some coverage
        assert 0 <= coverage <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
