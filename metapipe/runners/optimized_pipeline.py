#!/usr/bin/env python3
"""
Optimized MetaPipe Pipeline

Integrates Solutions 1 + 2 + 3 from LATENCY_ANALYSIS.md:
- TCARExtractorOptimized: Feature caching + Fast-TCAR
- FastMAPPolicy: Smaller network + INT8 quantization
- Maintains BCPR and UQE (low overhead)

Expected performance:
- Latency: 992ms → 972ms (-20ms, -2.0%)
- Quality: 8.97 → 9.15 (-2% degradation, still competitive)
- Cost: No change (0.247)
"""

from __future__ import annotations
import numpy as np
import time
from typing import Dict, List, Optional, Any

from ..features.tcar_optimized import TCARExtractorOptimized
from ..policy.map_optimized import FastMAPPolicy
from ..optimizer.bcpr import BCPROptimizer
from ..uncertainty.uqe import ConformalEscalation
from .timeseries_pipeline import TimeSeriesPipeline, PipelineConfig, PipelineResult


class OptimizedPipelineConfig(PipelineConfig):
    """
    Configuration for optimized pipeline

    Extends base config with optimization flags
    """
    # Feature extraction optimizations
    cache_size: int = 1000  # Feature cache size
    cache_key_length: int = 50  # Hash key length
    acf_lags: int = 5  # Reduced from 20
    n_fft_features: int = 3  # Reduced from 10

    # Policy optimizations
    horizons: List[int] = [1, 6, 24]  # Reduced from [1,3,6,12,24]
    use_quantization: bool = True  # Enable INT8 quantization

    # Keep other parameters same as base
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

    cost_budget: float = 1.0
    latency_budget: float = 5000.0
    quality_threshold: float = 0.0

    escalation_enabled: bool = True
    escalation_alpha: float = 0.1

    transfer_enabled: bool = False
    n_source_domains: int = 20


class OptimizedMetaPipe(TimeSeriesPipeline):
    """
    Latency-Optimized MetaPipe Pipeline

    Combines three optimization strategies:
    1. Feature Caching (TCARExtractorOptimized)
    2. Fast-TCAR (reduced ACF/FFT dimensions)
    3. Fast-MAP (smaller network + quantization)

    Expected improvements:
    - Latency: -20ms (-2.0%)
    - Quality: -0.18 SMAPE (-2.0% degradation)
    - Cost: No change

    Trade-off: Slight quality degradation for significant latency improvement
    """

    def __init__(
        self,
        config: OptimizedPipelineConfig,
        model_registry: Dict[str, Any],
        use_meta_router: bool = False
    ):
        """
        Parameters
        ----------
        config : OptimizedPipelineConfig
            Optimized pipeline configuration
        model_registry : dict
            Registry of available models
        use_meta_router : bool
            If True, use meta-learned router (CPTL)
        """
        # Store config before parent init
        self.opt_config = config
        self.config = config  # Add for compatibility with parent class

        # Initialize optimized feature extractor
        self.feature_extractor = TCARExtractorOptimized(
            cache_size=config.cache_size,
            cache_key_length=config.cache_key_length,
            acf_lags=config.acf_lags,
            n_fft_features=config.n_fft_features
        )

        # Get feature dimension from optimized extractor
        dummy_features = self.feature_extractor.extract(
            np.random.randn(100),
            metadata={}
        )
        feature_dim = len(dummy_features.concat())

        # Initialize optimized routing policy
        n_actions = len(model_registry.get('forecast', {}))

        if use_meta_router and config.transfer_enabled:
            # TODO: Implement optimized MetaRouter
            from ..transfer.cptl import MetaRouter
            self.router = MetaRouter(
                feature_dim=feature_dim,
                n_actions=n_actions,
                n_source_domains=config.n_source_domains
            )
        else:
            # Use Fast-MAP
            self.router = FastMAPPolicy(
                feature_dim=feature_dim,
                n_actions=n_actions,
                horizons=config.horizons,
                learning_rate=config.learning_rate,
                gamma=config.gamma,
                use_quantization=config.use_quantization
            )

            # Apply quantization if enabled
            if config.use_quantization:
                self.router.quantize_model()

        # Initialize other components (BCPR, UQE) - keep original
        self.model_registry = model_registry

        self.budget_optimizer = BCPROptimizer(
            cost_budget=config.cost_budget,
            latency_budget=config.latency_budget,
            quality_threshold=config.quality_threshold
        )

        self.escalator = ConformalEscalation(
            alpha=config.escalation_alpha
        )

        # State
        self.epsilon = config.epsilon_start
        self.step_count = 0

        # Optimization statistics
        self.optimization_stats = {
            'tcar_cache_hits': 0,
            'tcar_cache_misses': 0,
            'avg_latency_ms': 0.0,
            'latency_samples': []
        }

    def run(
        self,
        x: np.ndarray,
        horizon: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
        budget_remaining: float = None
    ) -> PipelineResult:
        """
        Execute optimized pipeline

        Same interface as parent, but with latency tracking

        Parameters
        ----------
        x : np.ndarray
            Time series input
        horizon : int
            Forecasting horizon
        metadata : dict, optional
            Additional metadata
        budget_remaining : float, optional
            Remaining budget

        Returns
        -------
        PipelineResult
            Prediction with latency metrics
        """
        start_time = time.time()

        # Call parent run method
        result = super().run(x, horizon, metadata, budget_remaining)

        # Track latency
        total_latency_ms = (time.time() - start_time) * 1000
        self.optimization_stats['latency_samples'].append(total_latency_ms)

        # Update average latency
        samples = self.optimization_stats['latency_samples']
        self.optimization_stats['avg_latency_ms'] = np.mean(samples)

        # Update cache statistics
        cache_stats = self.feature_extractor.get_cache_stats()
        self.optimization_stats['tcar_cache_hits'] = cache_stats['hits']
        self.optimization_stats['tcar_cache_misses'] = cache_stats['misses']

        # Add optimization info to result
        if result.features is None:
            result.features = {}
        result.features['cache_hit_rate'] = cache_stats['hit_rate']

        return result

    def get_optimization_report(self) -> Dict[str, Any]:
        """
        Get detailed optimization statistics

        Returns
        -------
        dict
            Optimization metrics and cache statistics
        """
        # TCAR cache stats
        cache_stats = self.feature_extractor.get_cache_stats()

        # MAP model size
        map_size = self.router.get_model_size()

        # Latency statistics
        latencies = self.optimization_stats['latency_samples']
        if latencies:
            latency_stats = {
                'mean_ms': np.mean(latencies),
                'median_ms': np.median(latencies),
                'std_ms': np.std(latencies),
                'min_ms': np.min(latencies),
                'max_ms': np.max(latencies),
                'p95_ms': np.percentile(latencies, 95),
                'n_samples': len(latencies)
            }
        else:
            latency_stats = {}

        return {
            'tcar_cache': cache_stats,
            'map_model': map_size,
            'latency': latency_stats,
            'config': {
                'acf_lags': self.opt_config.acf_lags,
                'n_fft_features': self.opt_config.n_fft_features,
                'horizons': self.opt_config.horizons,
                'use_quantization': self.opt_config.use_quantization,
                'cache_size': self.opt_config.cache_size
            }
        }

    def reset_optimization_stats(self):
        """Reset optimization statistics"""
        self.optimization_stats = {
            'tcar_cache_hits': 0,
            'tcar_cache_misses': 0,
            'avg_latency_ms': 0.0,
            'latency_samples': []
        }
        self.feature_extractor.reset_cache()


def create_optimized_pipeline(
    model_registry: Dict[str, Any],
    cost_budget: float = 1.0,
    latency_budget: float = 5000.0,
    use_quantization: bool = True
) -> OptimizedMetaPipe:
    """
    Factory function to create optimized pipeline

    Parameters
    ----------
    model_registry : dict
        Available models
    cost_budget : float
        Cost budget
    latency_budget : float
        Latency budget in ms
    use_quantization : bool
        Enable INT8 quantization

    Returns
    -------
    OptimizedMetaPipe
        Configured pipeline
    """
    config = OptimizedPipelineConfig()
    config.cost_budget = cost_budget
    config.latency_budget = latency_budget
    config.use_quantization = use_quantization

    return OptimizedMetaPipe(
        config=config,
        model_registry=model_registry,
        use_meta_router=False
    )
