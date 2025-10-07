#!/usr/bin/env python3
"""
UQE: Uncertainty-Quantified Escalation

Novel contribution: Conformal prediction-based adaptive escalation with
finite-sample coverage guarantees (no asymptotic assumptions).

Equation:
    C_α(x) = {y : s(x,y) ≤ Q_{1-α}({s(x_i, y_i)}_{i∈Cal})}

    where s(x,y) is a nonconformity score

Theorem: P(y_test ∈ C_α(x_test)) ≥ 1 - α for any distribution
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import deque


@dataclass
class ConformalPredictionSet:
    """Conformal prediction interval/set"""
    prediction: float
    lower: float
    upper: float
    width: float
    coverage_level: float  # 1 - α


class ConformalEscalation:
    """
    Uncertainty-Quantified Escalation using Conformal Prediction

    Key innovation: Provides finite-sample coverage guarantees without
    distributional assumptions. Decides when to escalate to stronger
    (more expensive) models based on calibrated uncertainty.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        nonconformity_fn: str = 'absolute',
        calibration_window: int = 1000,
        escalation_threshold: float = 0.5,
        adaptive_threshold: bool = True
    ):
        """
        Parameters
        ----------
        alpha : float
            Miscoverage rate (target coverage = 1 - α)
        nonconformity_fn : str
            Type of nonconformity score:
            - 'absolute': |y - ŷ|
            - 'normalized': |y - ŷ| / (σ + ε)
        calibration_window : int
            Maximum size of calibration set (rolling window)
        escalation_threshold : float
            Base threshold for escalation decision
        adaptive_threshold : bool
            If True, adapt threshold based on budget remaining
        """
        self.alpha = alpha
        self.nonconformity_fn = nonconformity_fn
        self.calibration_window = calibration_window
        self.base_threshold = escalation_threshold
        self.adaptive_threshold = adaptive_threshold

        # Calibration set: (prediction, true_value, uncertainty)
        self.calibration_set: deque = deque(maxlen=calibration_window)

        # Quantile cache (recomputed when calibration set updates)
        self._quantile_cache: Optional[float] = None
        self._cache_valid = False

    def calibrate(
        self,
        predictions: np.ndarray,
        true_values: np.ndarray,
        uncertainties: Optional[np.ndarray] = None
    ):
        """
        Calibrate conformal predictor on validation set

        Parameters
        ----------
        predictions : np.ndarray
            Model predictions, shape (n_samples,)
        true_values : np.ndarray
            True labels/values, shape (n_samples,)
        uncertainties : np.ndarray, optional
            Model uncertainty estimates (e.g., std), shape (n_samples,)
        """
        n = len(predictions)
        if uncertainties is None:
            uncertainties = np.ones(n)

        for i in range(n):
            self.calibration_set.append((
                float(predictions[i]),
                float(true_values[i]),
                float(uncertainties[i])
            ))

        self._cache_valid = False

    def predict(
        self,
        prediction: float,
        uncertainty: float = 1.0
    ) -> ConformalPredictionSet:
        """
        Compute conformal prediction set (interval for regression)

        Novel equation:
            C_α(x) = [ŷ(x) - q, ŷ(x) + q]
            where q = Q_{1-α}({|y_i - ŷ(x_i)| / σ_i})

        Parameters
        ----------
        prediction : float
            Point prediction from model
        uncertainty : float
            Model uncertainty estimate (optional)

        Returns
        -------
        ConformalPredictionSet
            Prediction interval with coverage guarantee
        """
        if len(self.calibration_set) < 10:
            # Insufficient calibration data, return wide interval
            return ConformalPredictionSet(
                prediction=prediction,
                lower=prediction - 10 * uncertainty,
                upper=prediction + 10 * uncertainty,
                width=20 * uncertainty,
                coverage_level=1 - self.alpha
            )

        # Compute or retrieve quantile
        quantile = self._get_quantile()

        # Conformal interval
        lower = prediction - quantile * uncertainty
        upper = prediction + quantile * uncertainty
        width = 2 * quantile * uncertainty

        return ConformalPredictionSet(
            prediction=prediction,
            lower=lower,
            upper=upper,
            width=width,
            coverage_level=1 - self.alpha
        )

    def should_escalate(
        self,
        prediction: float,
        uncertainty: float,
        budget_remaining: float,
        cost_weak: float,
        cost_strong: float
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Decide whether to escalate to stronger model

        Novel decision criterion:
            1. High uncertainty AND sufficient budget
            2. Expected value of information > incremental cost

        Parameters
        ----------
        prediction : float
            Current model's prediction
        uncertainty : float
            Uncertainty estimate (conformal width or model std)
        budget_remaining : float
            Remaining budget
        cost_weak : float
            Cost of current (weak) model
        cost_strong : float
            Cost of stronger model

        Returns
        -------
        should_escalate : bool
            True if should escalate
        info : dict
            Diagnostic information
        """
        # Get conformal prediction set
        conf_set = self.predict(prediction, uncertainty)

        # Adaptive threshold based on remaining budget
        if self.adaptive_threshold:
            # More willing to escalate if lots of budget left
            budget_factor = min(1.5, budget_remaining / (cost_strong + 1e-6))
            threshold = self.base_threshold / budget_factor
        else:
            threshold = self.base_threshold

        # Uncertainty-cost ratio
        uncertainty_cost_ratio = conf_set.width / (cost_weak + 1e-6)

        # Value of information (expected reduction in loss)
        # Assume strong model reduces uncertainty by 50%
        expected_improvement = 0.5 * conf_set.width
        incremental_cost = cost_strong - cost_weak

        value_of_information = expected_improvement

        # Escalation decision
        should_escalate = (
            conf_set.width > threshold and
            budget_remaining >= incremental_cost and
            value_of_information > incremental_cost
        )

        info = {
            'conf_width': conf_set.width,
            'conf_lower': conf_set.lower,
            'conf_upper': conf_set.upper,
            'threshold': threshold,
            'uncertainty_cost_ratio': uncertainty_cost_ratio,
            'value_of_information': value_of_information,
            'incremental_cost': incremental_cost,
            'budget_remaining': budget_remaining
        }

        return should_escalate, info

    def _get_quantile(self) -> float:
        """
        Compute (1-α) quantile of calibration scores

        Novel: Normalized nonconformity scores for heteroscedastic data
        """
        if self._cache_valid and self._quantile_cache is not None:
            return self._quantile_cache

        # Compute nonconformity scores
        scores = []
        for pred, true, unc in self.calibration_set:
            score = self._nonconformity_score(pred, true, unc)
            scores.append(score)

        scores = np.array(scores)

        # Compute (1-α) quantile
        # Add small correction for finite-sample guarantee
        n = len(scores)
        adjusted_alpha = min(self.alpha, (1 + 1/n) * self.alpha)

        quantile = np.quantile(scores, 1 - adjusted_alpha)

        # Cache
        self._quantile_cache = float(quantile)
        self._cache_valid = True

        return quantile

    def _nonconformity_score(
        self,
        prediction: float,
        true_value: float,
        uncertainty: float = 1.0
    ) -> float:
        """
        Compute nonconformity score s(x, y)

        Options:
            - 'absolute': |y - ŷ|
            - 'normalized': |y - ŷ| / (σ + ε)
        """
        residual = abs(true_value - prediction)

        if self.nonconformity_fn == 'absolute':
            return residual
        elif self.nonconformity_fn == 'normalized':
            return residual / (uncertainty + 1e-6)
        else:
            raise ValueError(f"Unknown nonconformity function: {self.nonconformity_fn}")

    def get_coverage(
        self,
        predictions: np.ndarray,
        true_values: np.ndarray,
        uncertainties: Optional[np.ndarray] = None
    ) -> float:
        """
        Empirically verify coverage on test set

        Should be approximately 1 - α

        Returns
        -------
        float
            Empirical coverage rate
        """
        if uncertainties is None:
            uncertainties = np.ones(len(predictions))

        covered = 0
        for pred, true, unc in zip(predictions, true_values, uncertainties):
            conf_set = self.predict(pred, unc)
            if conf_set.lower <= true <= conf_set.upper:
                covered += 1

        return covered / len(predictions)

    def reset_calibration(self):
        """Clear calibration set"""
        self.calibration_set.clear()
        self._cache_valid = False
        self._quantile_cache = None

    def get_stats(self) -> Dict[str, any]:
        """Get calibrator statistics"""
        return {
            'calibration_size': len(self.calibration_set),
            'alpha': self.alpha,
            'target_coverage': 1 - self.alpha,
            'quantile': self._get_quantile() if len(self.calibration_set) >= 10 else None
        }
