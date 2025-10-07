#!/usr/bin/env python3
"""
Tests for UQE (Uncertainty-Quantified Escalation)
"""

import pytest
import numpy as np
from metapipe.uncertainty.uqe import ConformalEscalation, ConformalPredictionSet


class TestConformalEscalation:
    """Test suite for Conformal Escalation"""

    @pytest.fixture
    def escalator(self):
        """Create conformal escalation instance"""
        return ConformalEscalation(
            alpha=0.1,
            nonconformity_fn='absolute',
            calibration_window=100
        )

    def test_initialization(self, escalator):
        """Test escalator initialization"""
        assert escalator.alpha == 0.1
        assert len(escalator.calibration_set) == 0

    def test_calibration(self, escalator):
        """Test calibration with validation set"""
        n = 50
        predictions = np.random.randn(n)
        true_values = predictions + np.random.randn(n) * 0.5

        escalator.calibrate(predictions, true_values)

        assert len(escalator.calibration_set) == n

    def test_predict_without_calibration(self, escalator):
        """Test prediction without calibration returns wide interval"""
        conf_set = escalator.predict(prediction=10.0, uncertainty=1.0)

        assert isinstance(conf_set, ConformalPredictionSet)
        assert conf_set.width > 5.0  # should be wide

    def test_predict_with_calibration(self, escalator):
        """Test prediction with calibration"""
        # Calibrate
        predictions = np.random.randn(100) * 2 + 10
        true_values = predictions + np.random.randn(100) * 0.5
        escalator.calibrate(predictions, true_values)

        # Predict
        conf_set = escalator.predict(prediction=10.0, uncertainty=1.0)

        assert conf_set.prediction == 10.0
        assert conf_set.lower < conf_set.upper
        assert conf_set.width == conf_set.upper - conf_set.lower
        assert conf_set.coverage_level == 0.9  # 1 - alpha

    def test_coverage_guarantee(self, escalator):
        """Test empirical coverage matches theoretical guarantee"""
        # Calibration set
        np.random.seed(42)
        cal_preds = np.random.randn(100)
        cal_true = cal_preds + np.random.randn(100) * 0.5
        escalator.calibrate(cal_preds, cal_true)

        # Test set
        test_preds = np.random.randn(100)
        test_true = test_preds + np.random.randn(100) * 0.5

        coverage = escalator.get_coverage(test_preds, test_true)

        # Coverage should be approximately 1 - alpha = 0.9
        # Allow some slack due to finite sample
        assert 0.8 <= coverage <= 1.0

    def test_escalation_decision_high_uncertainty(self, escalator):
        """Test escalation with high uncertainty"""
        # Calibrate
        escalator.calibrate(np.random.randn(100), np.random.randn(100))

        # High uncertainty, sufficient budget
        should_escalate, info = escalator.should_escalate(
            prediction=10.0,
            uncertainty=5.0,  # high
            budget_remaining=1.0,
            cost_weak=0.1,
            cost_strong=0.3
        )

        # Might escalate due to high uncertainty
        assert isinstance(should_escalate, bool)
        assert 'conf_width' in info
        assert 'value_of_information' in info

    def test_escalation_decision_low_budget(self, escalator):
        """Test no escalation when budget is insufficient"""
        escalator.calibrate(np.random.randn(100), np.random.randn(100))

        should_escalate, info = escalator.should_escalate(
            prediction=10.0,
            uncertainty=1.0,
            budget_remaining=0.05,  # insufficient
            cost_weak=0.1,
            cost_strong=0.3
        )

        # Should not escalate due to insufficient budget
        assert should_escalate == False

    def test_normalized_nonconformity(self):
        """Test normalized nonconformity scores"""
        escalator = ConformalEscalation(
            alpha=0.1,
            nonconformity_fn='normalized'
        )

        score1 = escalator._nonconformity_score(10.0, 12.0, uncertainty=1.0)
        score2 = escalator._nonconformity_score(10.0, 12.0, uncertainty=2.0)

        # With same residual but different uncertainty, normalized scores differ
        assert score1 > score2  # smaller uncertainty → larger normalized score

    def test_adaptive_threshold(self):
        """Test adaptive threshold based on budget"""
        escalator = ConformalEscalation(
            alpha=0.1,
            adaptive_threshold=True,
            escalation_threshold=0.5
        )

        escalator.calibrate(np.random.randn(100), np.random.randn(100))

        # More budget → lower threshold (more willing to escalate)
        _, info1 = escalator.should_escalate(10.0, 1.0, budget_remaining=10.0, cost_weak=0.1, cost_strong=0.3)
        _, info2 = escalator.should_escalate(10.0, 1.0, budget_remaining=0.5, cost_weak=0.1, cost_strong=0.3)

        assert info1['threshold'] <= info2['threshold']

    def test_reset_calibration(self, escalator):
        """Test reset clears calibration"""
        escalator.calibrate(np.random.randn(50), np.random.randn(50))
        assert len(escalator.calibration_set) > 0

        escalator.reset_calibration()
        assert len(escalator.calibration_set) == 0


class TestConformalIntegration:
    """Integration tests for conformal prediction"""

    def test_sequential_calibration_updates(self):
        """Test that calibration updates incrementally"""
        escalator = ConformalEscalation(alpha=0.1, calibration_window=50)

        # Add in batches
        for i in range(3):
            preds = np.random.randn(20) + i
            trues = preds + np.random.randn(20) * 0.5
            escalator.calibrate(preds, trues)

        # Should maintain window size
        assert len(escalator.calibration_set) == 50

    def test_conformal_prediction_stability(self):
        """Test that predictions are stable across calls"""
        escalator = ConformalEscalation(alpha=0.1)
        escalator.calibrate(np.random.randn(100), np.random.randn(100))

        # Same input should give same output
        conf1 = escalator.predict(10.0, 1.0)
        conf2 = escalator.predict(10.0, 1.0)

        assert conf1.lower == conf2.lower
        assert conf1.upper == conf2.upper


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
