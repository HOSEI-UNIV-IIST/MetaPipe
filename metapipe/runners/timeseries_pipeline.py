#!/usr/bin/env python3
"""
Time-Series Pipeline with MetaPipe Routing

Integrates all MetaPipe components:
- TCAR feature extraction
- MAP policy for routing
- BCPR budget constraints
- UQE escalation
- CPTL transfer learning
"""

from __future__ import annotations
import numpy as np
import torch
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from ..features.tcar import TCARExtractor
from ..policy.map import MAPPolicy, HorizonReward
from ..optimizer.bcpr import BCPROptimizer
from ..uncertainty.uqe import ConformalEscalation
from ..transfer.cptl import MetaRouter


@dataclass
class PipelineConfig:
    """Configuration for time-series pipeline"""
    # Feature extraction
    acf_lags: int = 20
    pacf_lags: int = 20
    n_fft_features: int = 10

    # Policy
    horizons: List[int] = field(default_factory=lambda: [1, 3, 6, 12, 24])
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

    # Budget constraints
    cost_budget: float = 1.0
    latency_budget: float = 5000.0  # ms
    quality_threshold: float = 0.0

    # Escalation
    escalation_enabled: bool = True
    escalation_alpha: float = 0.1

    # Transfer learning
    transfer_enabled: bool = False
    n_source_domains: int = 20


@dataclass
class PipelineResult:
    """Result from pipeline execution"""
    prediction: np.ndarray
    uncertainty: float
    cost: float
    latency_ms: float
    quality: Optional[float] = None
    selected_models: Dict[str, str] = field(default_factory=dict)
    escalated: bool = False
    features: Optional[Dict[str, np.ndarray]] = None


class TimeSeriesPipeline:
    """
    MetaPipe Time-Series Pipeline

    Stages:
    1. Feature Extraction (TCAR)
    2. Model Selection (MAP policy)
    3. Forecasting/Detection
    4. Uncertainty Quantification (UQE)
    5. Optional Escalation
    """

    def __init__(
        self,
        config: PipelineConfig,
        model_registry: Dict[str, Any],
        use_meta_router: bool = False
    ):
        """
        Parameters
        ----------
        config : PipelineConfig
            Pipeline configuration
        model_registry : dict
            Registry of available models per stage
            {
                'forecast': {'arima': ..., 'lstm': ..., 'transformer': ...},
                'detect': {...},
                'validate': {...}
            }
        use_meta_router : bool
            If True, use meta-learned router (CPTL)
        """
        self.config = config
        self.model_registry = model_registry

        # Initialize components
        self.feature_extractor = TCARExtractor(
            acf_lags=config.acf_lags,
            pacf_lags=config.pacf_lags,
            n_fft_features=config.n_fft_features
        )

        # Determine feature dimension
        dummy_features = self.feature_extractor.extract(
            np.random.randn(100),
            metadata={}
        )
        feature_dim = len(dummy_features.concat())

        # Routing policy
        n_actions = len(model_registry.get('forecast', {}))

        if use_meta_router and config.transfer_enabled:
            self.router = MetaRouter(
                feature_dim=feature_dim,
                n_actions=n_actions,
                n_source_domains=config.n_source_domains
            )
        else:
            self.router = MAPPolicy(
                feature_dim=feature_dim,
                n_actions=n_actions,
                horizons=config.horizons,
                learning_rate=config.learning_rate,
                gamma=config.gamma
            )

        # Budget optimizer
        self.budget_optimizer = BCPROptimizer(
            cost_budget=config.cost_budget,
            latency_budget=config.latency_budget,
            quality_threshold=config.quality_threshold
        )

        # Uncertainty quantification
        self.escalator = ConformalEscalation(
            alpha=config.escalation_alpha
        )

        # State
        self.epsilon = config.epsilon_start
        self.step_count = 0

    def run(
        self,
        x: np.ndarray,
        horizon: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
        budget_remaining: float = None
    ) -> PipelineResult:
        """
        Execute pipeline on time series

        Parameters
        ----------
        x : np.ndarray
            Time series input, shape (seq_len,) or (seq_len, n_features)
        horizon : int
            Forecasting horizon
        metadata : dict, optional
            Additional metadata (domain, task type, etc.)
        budget_remaining : float, optional
            Remaining budget for this episode

        Returns
        -------
        PipelineResult
        """
        start_time = time.time()

        metadata = metadata or {}
        metadata['horizon'] = horizon

        if budget_remaining is None:
            budget_remaining = self.config.cost_budget

        # Stage 1: Feature Extraction (TCAR)
        features = self.feature_extractor.extract(x, metadata)
        features_concat = features.concat()

        # Stage 2: Model Selection (MAP/MetaRouter)
        action = self.router.select_action(
            features_concat,
            epsilon=self.epsilon
        )

        # Map action to model
        forecast_models = list(self.model_registry.get('forecast', {}).keys())
        selected_model = forecast_models[action % len(forecast_models)]

        # Stage 3: Forecasting
        model = self.model_registry['forecast'][selected_model]
        prediction, model_cost, model_latency = self._run_model(
            model, x, horizon
        )

        # Stage 4: Uncertainty Quantification
        uncertainty = self._compute_uncertainty(prediction, model)

        # Stage 5: Escalation Decision
        escalated = False
        if self.config.escalation_enabled:
            should_escalate, esc_info = self.escalator.should_escalate(
                prediction=prediction.mean() if prediction.ndim > 0 else prediction,
                uncertainty=uncertainty,
                budget_remaining=budget_remaining,
                cost_weak=model_cost,
                cost_strong=model_cost * 2.0  # assume 2x cost for stronger model
            )

            if should_escalate:
                # Escalate to stronger model
                strong_model = self._get_stronger_model(selected_model)
                if strong_model is not None:
                    prediction, model_cost, model_latency = self._run_model(
                        strong_model, x, horizon
                    )
                    escalated = True
                    selected_model = f"{selected_model}_escalated"

        # Compute total latency
        total_latency = (time.time() - start_time) * 1000  # ms

        # Update budget optimizer
        # (quality unknown until we have ground truth)
        quality_placeholder = 0.5

        reward, violation = self.budget_optimizer.compute_augmented_reward(
            quality=quality_placeholder,
            cost=model_cost,
            latency=total_latency
        )

        self.budget_optimizer.update_dual_variables(violation)

        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )
        self.step_count += 1

        return PipelineResult(
            prediction=prediction,
            uncertainty=uncertainty,
            cost=model_cost,
            latency_ms=total_latency,
            quality=None,  # to be filled after evaluation
            selected_models={'forecast': selected_model},
            escalated=escalated,
            features=features.to_dict()
        )

    def update_policy(
        self,
        features: np.ndarray,
        action: int,
        rewards_per_horizon: List[HorizonReward],
        next_features: np.ndarray,
        done: bool = False
    ):
        """
        Update routing policy after receiving ground truth

        Parameters
        ----------
        features : np.ndarray
            State features
        action : int
            Action taken
        rewards_per_horizon : list
            Rewards for each horizon
        next_features : np.ndarray
            Next state features
        done : bool
            Episode termination
        """
        if isinstance(self.router, MAPPolicy):
            self.router.update(
                features, action, rewards_per_horizon, next_features, done
            )

    def calibrate_uncertainty(
        self,
        predictions: np.ndarray,
        true_values: np.ndarray
    ):
        """
        Calibrate conformal predictor on validation set

        Parameters
        ----------
        predictions : np.ndarray
            Model predictions
        true_values : np.ndarray
            Ground truth values
        """
        self.escalator.calibrate(predictions, true_values)

    def _run_model(
        self,
        model: Any,
        x: np.ndarray,
        horizon: int
    ) -> Tuple[np.ndarray, float, float]:
        """
        Run forecasting model

        Returns
        -------
        prediction : np.ndarray
        cost : float
        latency_ms : float
        """
        start = time.time()

        # Dummy implementation - replace with actual model call
        if callable(model):
            prediction = model(x, horizon=horizon)
        else:
            # Simple baseline: repeat last value
            prediction = np.repeat(x[-1], horizon)

        latency_ms = (time.time() - start) * 1000

        # Estimate cost (placeholder)
        cost = 0.1  # base cost

        return prediction, cost, latency_ms

    def _compute_uncertainty(
        self,
        prediction: np.ndarray,
        model: Any
    ) -> float:
        """
        Compute uncertainty estimate

        Options:
        - Model-based (e.g., ensemble variance)
        - Conformal prediction width
        - Heuristic based on prediction variance
        """
        if prediction.ndim > 0:
            # Use variance as uncertainty proxy
            uncertainty = float(np.std(prediction))
        else:
            uncertainty = 0.1  # default

        return uncertainty

    def _get_stronger_model(self, current_model: str) -> Optional[Any]:
        """
        Get a stronger (more expensive/accurate) model

        Simple heuristic: models ordered by cost/quality
        """
        forecast_models = list(self.model_registry.get('forecast', {}).keys())

        try:
            current_idx = forecast_models.index(current_model)
            if current_idx < len(forecast_models) - 1:
                next_model = forecast_models[current_idx + 1]
                return self.model_registry['forecast'][next_model]
        except (ValueError, IndexError):
            pass

        return None

    def save_checkpoint(self, path: str):
        """Save pipeline state"""
        self.router.save(path + '_router.pt')
        # Save other components as needed

    def load_checkpoint(self, path: str):
        """Load pipeline state"""
        self.router.load(path + '_router.pt')
