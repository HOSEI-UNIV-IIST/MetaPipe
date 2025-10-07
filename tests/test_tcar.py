#!/usr/bin/env python3
"""
Tests for TCAR (Temporal Context-Aware Routing) Feature Extraction
"""

import pytest
import numpy as np
from metapipe.features.tcar import TCARExtractor, TCARFeatures


class TestTCARExtractor:
    """Test suite for TCARExtractor"""

    @pytest.fixture
    def extractor(self):
        """Create TCAR extractor instance"""
        return TCARExtractor(
            acf_lags=20,
            pacf_lags=20,
            n_fft_features=10,
            seasonality_period=24
        )

    @pytest.fixture
    def sample_series(self):
        """Generate sample time series"""
        np.random.seed(42)
        t = np.linspace(0, 10, 200)
        # Trend + seasonality + noise
        series = 0.5 * t + 10 * np.sin(2 * np.pi * t / 5) + np.random.randn(200) * 0.5
        return series

    def test_extract_returns_tcar_features(self, extractor, sample_series):
        """Test that extract returns TCARFeatures object"""
        features = extractor.extract(sample_series)
        assert isinstance(features, TCARFeatures)

    def test_statistical_features_shape(self, extractor, sample_series):
        """Test statistical features have correct shape"""
        features = extractor.extract(sample_series)
        # Should have: mean, std, min, max, median, q25, q75, skew, kurtosis, cv = 10 features
        assert features.statistical.shape == (10,)

    def test_temporal_features_shape(self, extractor, sample_series):
        """Test temporal features have correct shape"""
        features = extractor.extract(sample_series)
        # ACF (20) + PACF (20) + trend_strength + seasonality_strength = 42
        expected_len = extractor.acf_lags + extractor.pacf_lags + 2
        assert features.temporal.shape == (expected_len,)

    def test_spectral_features_shape(self, extractor, sample_series):
        """Test spectral features have correct shape"""
        features = extractor.extract(sample_series)
        # top_k freqs (10) + top_k power (10) + spectral_entropy (1) = 21
        expected_len = extractor.n_fft_features * 2 + 1
        assert features.spectral.shape == (expected_len,)

    def test_meta_features_shape(self, extractor, sample_series):
        """Test meta features have correct shape"""
        metadata = {'domain': 'finance', 'task_type': 'forecast', 'horizon': 12}
        features = extractor.extract(sample_series, metadata)
        # seq_len, n_missing, missing_pct, horizon, domain_id, task_id = 6
        assert features.meta.shape == (6,)

    def test_concat_features(self, extractor, sample_series):
        """Test concatenated features"""
        features = extractor.extract(sample_series)
        concat = features.concat()
        # 10 + 42 + 21 + 6 = 79
        expected_len = 10 + 42 + 21 + 6
        assert concat.shape == (expected_len,)

    def test_temporal_similarity_kernel(self, extractor, sample_series):
        """Test temporal similarity kernel"""
        series2 = sample_series + np.random.randn(len(sample_series)) * 0.1
        similarity = extractor.temporal_similarity_kernel(sample_series, series2)

        # Similarity should be in [0, 1]
        assert 0 <= similarity <= 1

        # Similar series should have high similarity
        assert similarity > 0.5

    def test_trend_strength_positive_trend(self, extractor):
        """Test trend strength detection for positive trend"""
        t = np.linspace(0, 10, 100)
        series = 2 * t + np.random.randn(100) * 0.1  # strong trend

        trend_strength = extractor._compute_trend_strength(series)
        assert trend_strength > 0.8  # should detect strong trend

    def test_seasonality_strength_seasonal_series(self, extractor):
        """Test seasonality detection for seasonal series"""
        t = np.linspace(0, 10, 240)  # 10 periods of 24
        series = 10 * np.sin(2 * np.pi * t / 1.0) + np.random.randn(240) * 0.1

        seasonality_strength = extractor._compute_seasonality_strength(series)
        assert seasonality_strength > 0.5  # should detect seasonality

    def test_handles_nan_values(self, extractor):
        """Test handling of NaN values"""
        series = np.random.randn(100)
        series[50:55] = np.nan  # introduce missing values

        features = extractor.extract(series)
        # Should not crash and return valid features
        assert not np.any(np.isnan(features.statistical))

    def test_multivariate_series(self, extractor):
        """Test extraction from multivariate series"""
        series = np.random.randn(100, 3)  # 100 timesteps, 3 features
        features = extractor.extract(series)

        # Should extract from first dimension
        assert features.concat().shape[0] > 0

    def test_to_dict(self, extractor, sample_series):
        """Test to_dict conversion"""
        features = extractor.extract(sample_series)
        feature_dict = features.to_dict()

        assert 'statistical' in feature_dict
        assert 'temporal' in feature_dict
        assert 'spectral' in feature_dict
        assert 'meta' in feature_dict


class TestTCARIntegration:
    """Integration tests for TCAR"""

    def test_reproducibility(self):
        """Test that extraction is reproducible"""
        np.random.seed(42)
        series = np.random.randn(100)

        extractor1 = TCARExtractor()
        extractor2 = TCARExtractor()

        features1 = extractor1.extract(series)
        features2 = extractor2.extract(series)

        np.testing.assert_array_almost_equal(
            features1.concat(),
            features2.concat()
        )

    def test_different_series_different_features(self):
        """Test that different series produce different features"""
        np.random.seed(42)
        series1 = np.random.randn(100)
        series2 = np.random.randn(100) * 10 + 5  # different scale and offset

        extractor = TCARExtractor()
        features1 = extractor.extract(series1)
        features2 = extractor.extract(series2)

        # Features should be different
        assert not np.allclose(features1.concat(), features2.concat())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
