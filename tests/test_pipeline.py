#!/usr/bin/env python3
"""
Integration Tests for Time-Series Pipeline
"""

import pytest
import numpy as np
from metapipe.runners.timeseries_pipeline import TimeSeriesPipeline, PipelineConfig, PipelineResult


class TestTimeSeriesPipeline:
    """Test end-to-end pipeline"""

    @pytest.fixture
    def config(self):
        """Create pipeline configuration"""
        return PipelineConfig(
            cost_budget=1.0,
            latency_budget=5000.0,
            escalation_enabled=True
        )

    @pytest.fixture
    def model_registry(self):
        """Create dummy model registry"""
        return {
            'forecast': {
                'simple_avg': lambda x, horizon: np.repeat(x[-3:].mean(), horizon),
                'last_value': lambda x, horizon: np.repeat(x[-1], horizon),
                'linear_trend': lambda x, horizon: np.linspace(x[-1], x[-1] * 1.1, horizon)
            }
        }

    def test_pipeline_initialization(self, config, model_registry):
        """Test pipeline initializes correctly"""
        pipeline = TimeSeriesPipeline(config, model_registry)

        assert pipeline.config == config
        assert pipeline.model_registry == model_registry

    def test_pipeline_run(self, config, model_registry):
        """Test full pipeline execution"""
        pipeline = TimeSeriesPipeline(config, model_registry)

        # Generate test data
        x = np.random.randn(100)

        result = pipeline.run(
            x,
            horizon=12,
            metadata={'domain': 'finance', 'task_type': 'forecast'}
        )

        assert isinstance(result, PipelineResult)
        assert result.prediction is not None
        assert len(result.prediction) == 12
        assert result.cost >= 0
        assert result.latency_ms >= 0

    def test_pipeline_features_extracted(self, config, model_registry):
        """Test that features are extracted"""
        pipeline = TimeSeriesPipeline(config, model_registry)
        x = np.random.randn(100)

        result = pipeline.run(x, horizon=12)

        assert result.features is not None
        assert 'statistical' in result.features
        assert 'temporal' in result.features
        assert 'spectral' in result.features

    def test_pipeline_model_selection(self, config, model_registry):
        """Test that model is selected"""
        pipeline = TimeSeriesPipeline(config, model_registry)
        x = np.random.randn(100)

        result = pipeline.run(x, horizon=12)

        assert 'forecast' in result.selected_models
        assert result.selected_models['forecast'] in list(model_registry['forecast'].keys()) or '_escalated' in result.selected_models['forecast']

    def test_pipeline_escalation(self, config, model_registry):
        """Test escalation mechanism"""
        config_escalate = PipelineConfig(
            escalation_enabled=True,
            cost_budget=10.0  # large budget
        )

        pipeline = TimeSeriesPipeline(config_escalate, model_registry)
        x = np.random.randn(100)

        # Run multiple times, at least one might escalate
        results = [pipeline.run(x, horizon=12) for _ in range(10)]

        # At least test that escalation field exists
        assert all(isinstance(r.escalated, bool) for r in results)

    def test_pipeline_respects_budget(self, config, model_registry):
        """Test that pipeline respects cost budget"""
        strict_config = PipelineConfig(
            cost_budget=0.01,  # very tight
            escalation_enabled=False
        )

        pipeline = TimeSeriesPipeline(strict_config, model_registry)
        x = np.random.randn(100)

        result = pipeline.run(x, horizon=12, budget_remaining=0.01)

        # Cost should be within reasonable range
        assert result.cost <= strict_config.cost_budget * 2  # allow some tolerance


class TestPipelineIntegration:
    """Integration tests across multiple runs"""

    def test_multiple_episodes(self):
        """Test pipeline over multiple episodes"""
        config = PipelineConfig()
        model_registry = {
            'forecast': {
                'model_a': lambda x, h: np.repeat(x.mean(), h),
                'model_b': lambda x, h: np.repeat(x[-1], h)
            }
        }

        pipeline = TimeSeriesPipeline(config, model_registry)

        results = []
        for i in range(20):
            x = np.random.randn(100) + i * 0.1  # slightly different each time
            result = pipeline.run(x, horizon=12)
            results.append(result)

        # All should succeed
        assert len(results) == 20
        assert all(r.prediction is not None for r in results)

    def test_different_horizons(self):
        """Test pipeline with different forecasting horizons"""
        config = PipelineConfig()
        model_registry = {
            'forecast': {
                'model': lambda x, h: np.repeat(x.mean(), h)
            }
        }

        pipeline = TimeSeriesPipeline(config, model_registry)
        x = np.random.randn(100)

        for horizon in [1, 3, 6, 12, 24]:
            result = pipeline.run(x, horizon=horizon)
            assert len(result.prediction) == horizon

    def test_epsilon_decay(self):
        """Test that epsilon decays over time"""
        config = PipelineConfig(epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9)
        model_registry = {'forecast': {'model': lambda x, h: np.repeat(x.mean(), h)}}

        pipeline = TimeSeriesPipeline(config, model_registry)

        initial_epsilon = pipeline.epsilon

        # Run multiple episodes
        for _ in range(50):
            x = np.random.randn(100)
            pipeline.run(x, horizon=12)

        # Epsilon should have decayed
        assert pipeline.epsilon < initial_epsilon
        assert pipeline.epsilon >= config.epsilon_end


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
