#!/usr/bin/env python3
"""
TCAR: Temporal Context-Aware Routing Feature Extractor

Novel contribution: Multi-modal feature embedding combining statistical,
temporal, spectral, and meta features for time-series routing.

Equation:
    Φ_TCAR(x_t) = [Φ_stat(x), Φ_temp(x), Φ_spec(x), Φ_meta(x)]
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from scipy import stats, signal
from scipy.spatial.distance import euclidean
from statsmodels.tsa.stattools import acf, pacf


@dataclass
class TCARFeatures:
    """Container for TCAR feature vectors"""
    statistical: np.ndarray
    temporal: np.ndarray
    spectral: np.ndarray
    meta: np.ndarray

    def concat(self) -> np.ndarray:
        """Concatenate all features into single vector"""
        return np.concatenate([
            self.statistical,
            self.temporal,
            self.spectral,
            self.meta
        ])

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dictionary"""
        return {
            'statistical': self.statistical,
            'temporal': self.temporal,
            'spectral': self.spectral,
            'meta': self.meta
        }


class TCARExtractor:
    """
    Temporal Context-Aware Routing Feature Extractor

    Extracts four types of features from time-series data:
    1. Statistical: mean, std, skew, kurtosis, percentiles
    2. Temporal: ACF, PACF, trend strength, seasonality
    3. Spectral: FFT features, dominant frequencies
    4. Meta: dataset properties, horizon, domain encoding
    """

    def __init__(
        self,
        acf_lags: int = 20,
        pacf_lags: int = 20,
        n_fft_features: int = 10,
        seasonality_period: Optional[int] = None
    ):
        """
        Parameters
        ----------
        acf_lags : int
            Number of lags for autocorrelation function
        pacf_lags : int
            Number of lags for partial autocorrelation
        n_fft_features : int
            Number of FFT frequency features to extract
        seasonality_period : int, optional
            Known seasonality period (e.g., 24 for hourly, 7 for daily)
        """
        self.acf_lags = acf_lags
        self.pacf_lags = pacf_lags
        self.n_fft_features = n_fft_features
        self.seasonality_period = seasonality_period

    def extract(
        self,
        x: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TCARFeatures:
        """
        Extract all TCAR features from time series

        Parameters
        ----------
        x : np.ndarray
            Time series data, shape (seq_len,) or (seq_len, n_features)
        metadata : dict, optional
            Additional metadata (domain, horizon, etc.)

        Returns
        -------
        TCARFeatures
            Extracted features
        """
        # Handle multivariate case (take first dimension)
        if x.ndim > 1:
            x = x[:, 0]

        metadata = metadata or {}

        return TCARFeatures(
            statistical=self.extract_statistical(x),
            temporal=self.extract_temporal(x),
            spectral=self.extract_spectral(x),
            meta=self.extract_meta(x, metadata)
        )

    def extract_statistical(self, x: np.ndarray) -> np.ndarray:
        """
        Extract statistical features

        Returns
        -------
        np.ndarray
            [mean, std, min, max, median, q25, q75, skew, kurtosis, cv]
        """
        features = [
            np.mean(x),
            np.std(x),
            np.min(x),
            np.max(x),
            np.median(x),
            np.percentile(x, 25),
            np.percentile(x, 75),
            stats.skew(x),
            stats.kurtosis(x),
            np.std(x) / (np.mean(x) + 1e-8)  # coefficient of variation
        ]
        return np.array(features, dtype=np.float32)

    def extract_temporal(self, x: np.ndarray) -> np.ndarray:
        """
        Extract temporal features (ACF, PACF, trend, seasonality)

        Novel contribution: Trend strength and seasonality strength metrics

        Returns
        -------
        np.ndarray
            [acf_1, ..., acf_k, pacf_1, ..., pacf_k, trend_strength, seasonality_strength]
        """
        # Autocorrelation
        try:
            acf_vals = acf(x, nlags=self.acf_lags, fft=True)[1:]  # exclude lag 0
        except Exception:
            acf_vals = np.zeros(self.acf_lags)

        # Partial autocorrelation
        try:
            pacf_vals = pacf(x, nlags=self.pacf_lags)[1:]  # exclude lag 0
        except Exception:
            pacf_vals = np.zeros(self.pacf_lags)

        # Trend strength (using STL decomposition approximation)
        trend_strength = self._compute_trend_strength(x)

        # Seasonality strength
        seasonality_strength = self._compute_seasonality_strength(x)

        features = np.concatenate([
            acf_vals,
            pacf_vals,
            [trend_strength, seasonality_strength]
        ])

        return features.astype(np.float32)

    def extract_spectral(self, x: np.ndarray) -> np.ndarray:
        """
        Extract spectral features via FFT

        Novel contribution: Dominant frequency ratios and power concentration

        Returns
        -------
        np.ndarray
            [freq_1, ..., freq_k, power_1, ..., power_k, spectral_entropy]
        """
        # Compute FFT
        fft_vals = np.fft.fft(x)
        freqs = np.fft.fftfreq(len(x))
        power = np.abs(fft_vals) ** 2

        # Get positive frequencies only
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        power = power[pos_mask]

        # Sort by power
        sorted_idx = np.argsort(power)[::-1]
        top_k = min(self.n_fft_features, len(freqs))

        # Top k frequencies and their power
        top_freqs = freqs[sorted_idx[:top_k]]
        top_power = power[sorted_idx[:top_k]]

        # Normalize power
        top_power_norm = top_power / (np.sum(power) + 1e-8)

        # Spectral entropy (measure of frequency diversity)
        spectral_entropy = -np.sum(
            top_power_norm * np.log(top_power_norm + 1e-8)
        )

        # Pad if necessary
        if len(top_freqs) < self.n_fft_features:
            top_freqs = np.pad(
                top_freqs,
                (0, self.n_fft_features - len(top_freqs)),
                mode='constant'
            )
            top_power_norm = np.pad(
                top_power_norm,
                (0, self.n_fft_features - len(top_power_norm)),
                mode='constant'
            )

        features = np.concatenate([
            top_freqs,
            top_power_norm,
            [spectral_entropy]
        ])

        return features.astype(np.float32)

    def extract_meta(
        self,
        x: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """
        Extract meta features from dataset properties

        Returns
        -------
        np.ndarray
            [seq_len, n_missing, missing_pct, horizon, domain_encoding]
        """
        seq_len = len(x)
        n_missing = np.sum(np.isnan(x))
        missing_pct = n_missing / seq_len

        # Horizon (from metadata)
        horizon = metadata.get('horizon', 1)

        # Domain encoding (one-hot or embedding index)
        domain_map = {
            'finance': 0, 'healthcare': 1, 'energy': 2,
            'iot': 3, 'traffic': 4, 'climate': 5, 'other': 6
        }
        domain = metadata.get('domain', 'other')
        domain_id = domain_map.get(domain, 6)

        # Task type encoding
        task_map = {'forecast': 0, 'classify': 1, 'detect': 2}
        task_type = metadata.get('task_type', 'forecast')
        task_id = task_map.get(task_type, 0)

        features = np.array([
            seq_len,
            n_missing,
            missing_pct,
            horizon,
            domain_id,
            task_id
        ], dtype=np.float32)

        return features

    def temporal_similarity_kernel(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        gamma_stat: float = 1.0,
        gamma_dtw: float = 0.1,
        gamma_spec: float = 1.0
    ) -> float:
        """
        Novel temporal similarity kernel combining three distances

        Equation:
            K(x_i, x_j) = exp(-γ₁||Φ_stat(x_i) - Φ_stat(x_j)||²)
                        × exp(-γ₂ DTW(x_i, x_j))
                        × exp(-γ₃ ||FFT(x_i) - FFT(x_j)||²)

        Parameters
        ----------
        x1, x2 : np.ndarray
            Time series to compare
        gamma_stat : float
            Weight for statistical distance
        gamma_dtw : float
            Weight for DTW distance
        gamma_spec : float
            Weight for spectral distance

        Returns
        -------
        float
            Similarity score in [0, 1]
        """
        # Statistical similarity
        stat1 = self.extract_statistical(x1)
        stat2 = self.extract_statistical(x2)
        stat_dist = np.sum((stat1 - stat2) ** 2)
        stat_sim = np.exp(-gamma_stat * stat_dist)

        # DTW similarity (approximate with Euclidean for speed)
        # TODO: Use proper DTW if needed
        dtw_dist = euclidean(x1, x2)
        dtw_sim = np.exp(-gamma_dtw * dtw_dist)

        # Spectral similarity
        spec1 = self.extract_spectral(x1)
        spec2 = self.extract_spectral(x2)
        spec_dist = np.sum((spec1 - spec2) ** 2)
        spec_sim = np.exp(-gamma_spec * spec_dist)

        # Combined similarity
        return stat_sim * dtw_sim * spec_sim

    def _compute_trend_strength(self, x: np.ndarray) -> float:
        """
        Compute trend strength using detrending

        Returns value in [0, 1] where 1 = strong trend
        """
        try:
            # Fit linear trend
            t = np.arange(len(x))
            coeffs = np.polyfit(t, x, deg=1)
            trend = np.polyval(coeffs, t)

            # Detrended residual
            detrended = x - trend

            # Trend strength = 1 - Var(detrended) / Var(x)
            var_x = np.var(x)
            var_detrended = np.var(detrended)

            if var_x < 1e-8:
                return 0.0

            strength = max(0.0, 1.0 - var_detrended / var_x)
            return float(strength)

        except Exception:
            return 0.0

    def _compute_seasonality_strength(self, x: np.ndarray) -> float:
        """
        Compute seasonality strength

        Returns value in [0, 1] where 1 = strong seasonality
        """
        if self.seasonality_period is None or len(x) < 2 * self.seasonality_period:
            return 0.0

        try:
            # Simple seasonal decomposition
            period = self.seasonality_period
            n_periods = len(x) // period

            # Reshape into periods
            trimmed = x[:n_periods * period]
            reshaped = trimmed.reshape(n_periods, period)

            # Seasonal component = mean across periods
            seasonal = np.mean(reshaped, axis=0)
            seasonal_full = np.tile(seasonal, n_periods)

            # Deseasonalized
            deseasonal = trimmed - seasonal_full

            # Seasonality strength
            var_x = np.var(trimmed)
            var_deseasonal = np.var(deseasonal)

            if var_x < 1e-8:
                return 0.0

            strength = max(0.0, 1.0 - var_deseasonal / var_x)
            return float(strength)

        except Exception:
            return 0.0
