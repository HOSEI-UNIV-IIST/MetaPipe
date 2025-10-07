#!/usr/bin/env python3
"""
Optimized TCAR: Feature Caching + Fast Feature Extraction

Implements Solutions 1 & 2 from LATENCY_ANALYSIS.md:
- Solution 1: Feature caching with LRU eviction (-7ms)
- Solution 2: Fast-TCAR with reduced ACF/FFT features (-5ms)

Expected latency reduction: 12ms total
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional
from collections import OrderedDict
import hashlib

from .tcar import TCARExtractor, TCARFeatures


class FastTCARExtractor(TCARExtractor):
    """
    Fast TCAR with reduced feature dimensions

    Optimizations:
    - ACF lags: 20 → 5 (4x speedup)
    - PACF: Removed (often redundant with ACF)
    - FFT features: 10 → 3 (3x speedup)
    - Fast ACF using FFT convolution instead of statsmodels

    Expected speedup: 3-5ms per extraction
    """

    def __init__(
        self,
        acf_lags: int = 5,  # Reduced from 20
        n_fft_features: int = 3,  # Reduced from 10
        seasonality_period: Optional[int] = None
    ):
        """
        Parameters
        ----------
        acf_lags : int
            Number of ACF lags (default: 5, reduced from 20)
        n_fft_features : int
            Number of FFT features (default: 3, reduced from 10)
        seasonality_period : int, optional
            Seasonality period
        """
        # Initialize with reduced parameters
        super().__init__(
            acf_lags=acf_lags,
            pacf_lags=0,  # Skip PACF
            n_fft_features=n_fft_features,
            seasonality_period=seasonality_period
        )

    def extract_temporal(self, x: np.ndarray) -> np.ndarray:
        """
        Fast temporal feature extraction

        Changes from parent:
        - Use FFT-based ACF (2-3x faster than statsmodels)
        - Skip PACF computation
        - Simplified trend/seasonality

        Returns
        -------
        np.ndarray
            [acf_1, ..., acf_k, trend_strength, seasonality_strength]
        """
        # Fast ACF using FFT convolution
        try:
            acf_vals = self._fast_acf(x, nlags=self.acf_lags)
        except Exception:
            acf_vals = np.zeros(self.acf_lags)

        # Simplified trend strength (no polynomial fit)
        trend_strength = self._fast_trend_strength(x)

        # Simplified seasonality (or skip if not needed)
        seasonality_strength = self._compute_seasonality_strength(x)

        features = np.concatenate([
            acf_vals,
            [trend_strength, seasonality_strength]
        ])

        return features.astype(np.float32)

    def _fast_acf(self, x: np.ndarray, nlags: int) -> np.ndarray:
        """
        Fast ACF using FFT convolution

        2-3x faster than statsmodels.tsa.stattools.acf

        Parameters
        ----------
        x : np.ndarray
            Input time series
        nlags : int
            Number of lags

        Returns
        -------
        np.ndarray
            ACF values at lags 1 to nlags
        """
        x_centered = x - x.mean()

        # FFT-based autocorrelation
        fft_x = np.fft.fft(x_centered, n=2*len(x))
        acf_full = np.fft.ifft(fft_x * np.conj(fft_x)).real[:nlags+1]

        # Normalize
        if acf_full[0] > 1e-8:
            acf_normalized = acf_full[1:] / acf_full[0]
        else:
            acf_normalized = np.zeros(nlags)

        return acf_normalized.astype(np.float32)

    def _fast_trend_strength(self, x: np.ndarray) -> float:
        """
        Fast trend strength using simple linear regression

        Faster than polynomial fitting in parent class

        Returns
        -------
        float
            Trend strength in [0, 1]
        """
        try:
            n = len(x)
            t = np.arange(n)

            # Simple linear regression (faster than polyfit)
            t_mean = t.mean()
            x_mean = x.mean()

            numerator = np.sum((t - t_mean) * (x - x_mean))
            denominator = np.sum((t - t_mean) ** 2)

            if denominator < 1e-8:
                return 0.0

            slope = numerator / denominator
            intercept = x_mean - slope * t_mean

            # Predicted values
            trend = slope * t + intercept

            # Trend strength
            var_x = np.var(x)
            var_residual = np.var(x - trend)

            if var_x < 1e-8:
                return 0.0

            strength = max(0.0, 1.0 - var_residual / var_x)
            return float(strength)

        except Exception:
            return 0.0


class TCARExtractorOptimized(FastTCARExtractor):
    """
    Optimized TCAR with feature caching

    Combines Fast-TCAR with LRU cache for repeated extractions

    Cache hit rate expected: 40-60% on validation/test sets
    Speedup on cache hit: ~7ms (entire TCAR extraction skipped)

    Memory overhead: ~1MB for 1000 cached features
    """

    def __init__(
        self,
        cache_size: int = 1000,
        cache_key_length: int = 50,
        **kwargs
    ):
        """
        Parameters
        ----------
        cache_size : int
            Maximum number of cached feature vectors (default: 1000)
        cache_key_length : int
            Number of trailing points to use for hash key (default: 50)
        **kwargs
            Passed to FastTCARExtractor
        """
        super().__init__(**kwargs)

        self.cache_size = cache_size
        self.cache_key_length = cache_key_length

        # LRU cache using OrderedDict
        self.feature_cache: OrderedDict[str, TCARFeatures] = OrderedDict()

        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0

    def extract(
        self,
        x: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TCARFeatures:
        """
        Extract features with caching

        Cache key: hash of last `cache_key_length` points

        Parameters
        ----------
        x : np.ndarray
            Time series data
        metadata : dict, optional
            Metadata

        Returns
        -------
        TCARFeatures
            Cached or newly computed features
        """
        # Handle multivariate case
        if x.ndim > 1:
            x_key = x[:, 0]
        else:
            x_key = x

        # Generate cache key from trailing points
        cache_key = self._compute_cache_key(x_key)

        # Check cache
        if cache_key in self.feature_cache:
            # Cache hit - move to end (LRU)
            self.feature_cache.move_to_end(cache_key)
            self.cache_hits += 1
            return self.feature_cache[cache_key]

        # Cache miss - compute features
        self.cache_misses += 1
        features = super().extract(x, metadata)

        # Add to cache
        self.feature_cache[cache_key] = features

        # Evict oldest if cache full
        if len(self.feature_cache) > self.cache_size:
            self.feature_cache.popitem(last=False)

        return features

    def _compute_cache_key(self, x: np.ndarray) -> str:
        """
        Compute cache key from time series

        Uses hash of last `cache_key_length` points for efficiency

        Parameters
        ----------
        x : np.ndarray
            Time series (1D)

        Returns
        -------
        str
            Hash key
        """
        # Use last N points for cache key
        n = min(self.cache_key_length, len(x))
        x_tail = x[-n:]

        # Hash using SHA256 (fast and collision-resistant)
        key_bytes = x_tail.tobytes()
        hash_obj = hashlib.sha256(key_bytes)
        return hash_obj.hexdigest()

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns
        -------
        dict
            Cache hit rate, size, etc.
        """
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0

        return {
            'cache_size': len(self.feature_cache),
            'max_size': self.cache_size,
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'total_queries': total,
            'hit_rate': hit_rate
        }

    def reset_cache(self):
        """Clear cache and statistics"""
        self.feature_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
