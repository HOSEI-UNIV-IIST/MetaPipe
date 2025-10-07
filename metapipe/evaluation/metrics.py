#!/usr/bin/env python3
"""
Evaluation Metrics for MetaPipe

Metrics:
- Quality: SMAPE, MASE, MAE, RMSE (forecasting)
- Quality: F1, Precision, Recall, AUC-ROC (anomaly detection/classification)
- Efficiency: Cost, Latency, Cost-Quality Ratio
- Adaptivity: Regret, Pareto Coverage, Transfer Efficiency
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    # Quality metrics
    smape: Optional[float] = None
    mase: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    f1: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    auc_roc: Optional[float] = None

    # Efficiency metrics
    cost: float = 0.0
    latency_ms: float = 0.0
    cost_quality_ratio: Optional[float] = None

    # Adaptivity metrics
    regret: Optional[float] = None
    pareto_coverage: Optional[float] = None


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error

    sMAPE = (100/n) * Σ |y_true - y_pred| / ((|y_true| + |y_pred|) / 2)
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return float(100 * np.mean(numerator / (denominator + 1e-8)))


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """
    Mean Absolute Scaled Error

    MASE = MAE(pred) / MAE(naive)
    where naive forecast = last observed value (for non-seasonal)
    """
    mae_pred = np.mean(np.abs(y_true - y_pred))

    # Naive forecast MAE on training set
    naive_errors = np.abs(np.diff(y_train))
    mae_naive = np.mean(naive_errors)

    return float(mae_pred / (mae_naive + 1e-8))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = 'forecast',
    y_train: Optional[np.ndarray] = None,
    cost: float = 0.0,
    latency_ms: float = 0.0
) -> EvaluationMetrics:
    """
    Compute all relevant metrics based on task type

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth
    y_pred : np.ndarray
        Predictions
    task_type : str
        'forecast', 'classify', or 'detect'
    y_train : np.ndarray, optional
        Training data for MASE computation
    cost : float
        Execution cost
    latency_ms : float
        Latency in milliseconds

    Returns
    -------
    EvaluationMetrics
    """
    metrics = EvaluationMetrics(cost=cost, latency_ms=latency_ms)

    if task_type == 'forecast':
        # Forecasting metrics
        metrics.smape = smape(y_true, y_pred)
        metrics.mae = mae(y_true, y_pred)
        metrics.rmse = rmse(y_true, y_pred)

        if y_train is not None:
            metrics.mase = mase(y_true, y_pred, y_train)

        # Cost-quality ratio
        if metrics.smape is not None and metrics.smape > 0:
            metrics.cost_quality_ratio = cost / metrics.smape

    elif task_type in ['classify', 'detect']:
        # Classification/detection metrics
        y_true_int = y_true.astype(int)
        y_pred_int = (y_pred > 0.5).astype(int)

        metrics.f1 = f1_score(y_true_int, y_pred_int, average='binary', zero_division=0)
        metrics.precision = precision_score(y_true_int, y_pred_int, average='binary', zero_division=0)
        metrics.recall = recall_score(y_true_int, y_pred_int, average='binary', zero_division=0)

        try:
            metrics.auc_roc = roc_auc_score(y_true_int, y_pred)
        except ValueError:
            metrics.auc_roc = None

        # Cost-quality ratio
        if metrics.f1 is not None and metrics.f1 > 0:
            metrics.cost_quality_ratio = cost / metrics.f1

    return metrics


def compute_regret(
    rewards: List[float],
    oracle_rewards: List[float]
) -> float:
    """
    Compute cumulative regret vs oracle

    Regret_T = Σ(r_oracle - r_policy)
    """
    regret = sum(o - r for o, r in zip(oracle_rewards, rewards))
    return float(regret)


def compute_pareto_coverage(
    obtained_points: List[tuple],
    oracle_pareto_frontier: List[tuple]
) -> float:
    """
    Compute what % of oracle Pareto frontier is covered

    Parameters
    ----------
    obtained_points : list of (quality, cost) tuples
        Points obtained by policy
    oracle_pareto_frontier : list of (quality, cost) tuples
        Oracle Pareto-optimal points

    Returns
    -------
    float
        Coverage ratio in [0, 1]
    """
    if not oracle_pareto_frontier:
        return 1.0

    covered = 0
    for oracle_point in oracle_pareto_frontier:
        # Check if any obtained point dominates or equals oracle point
        for obtained_point in obtained_points:
            if _dominates_or_equal(obtained_point, oracle_point):
                covered += 1
                break

    return covered / len(oracle_pareto_frontier)


def _dominates_or_equal(p1: tuple, p2: tuple) -> bool:
    """
    Check if p1 dominates or equals p2

    For (quality, cost): higher quality, lower cost is better
    """
    quality1, cost1 = p1
    quality2, cost2 = p2

    return quality1 >= quality2 and cost1 <= cost2
