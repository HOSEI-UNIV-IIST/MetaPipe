#!/usr/bin/env python3
"""
TSDB Unified Data Loader

Supports 30+ time-series datasets from public repositories:
- UCR Time Series Archive (128+ datasets)
- Monash Forecasting Repository (30+ datasets)
- PhysioNet (medical signals)
- Yahoo/Numenta (anomaly detection)
- UCI ML Repository (energy, air quality, traffic)
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod


@dataclass
class DatasetMetadata:
    """Metadata for time-series dataset"""
    name: str
    domain: str  # finance, healthcare, energy, iot, traffic, climate, other
    task_type: str  # forecast, classify, detect (anomaly)
    n_series: int
    seq_len: int
    n_features: int
    frequency: str  # hourly, daily, weekly, monthly, etc.
    has_missing: bool
    seasonality: Optional[int] = None
    horizon: int = 1  # forecasting horizon
    train_size: Optional[int] = None
    test_size: Optional[int] = None


@dataclass
class TimeSeriesData:
    """Container for time-series data"""
    X_train: np.ndarray  # (n_train, seq_len, n_features)
    y_train: np.ndarray  # targets
    X_test: np.ndarray   # (n_test, seq_len, n_features)
    y_test: np.ndarray
    metadata: DatasetMetadata

    def __post_init__(self):
        """Validate shapes"""
        assert self.X_train.ndim == 3, "X_train must be 3D: (n, seq_len, features)"
        assert self.X_test.ndim == 3, "X_test must be 3D"


class BaseLoader(ABC):
    """Abstract base class for dataset loaders"""

    @abstractmethod
    def load(self, dataset_name: str) -> TimeSeriesData:
        """Load dataset by name"""
        pass

    @abstractmethod
    def list_datasets(self) -> List[str]:
        """List available datasets"""
        pass


class TSDBDataset:
    """
    Unified interface for time-series databases

    Automatically routes to appropriate loader based on dataset name
    """

    # Registry of dataset name -> (loader_class, domain, task_type)
    DATASET_REGISTRY = {
        # Finance
        'stock_sp500': ('MonashLoader', 'finance', 'forecast'),
        'bitcoin': ('CryptoLoader', 'finance', 'forecast'),
        'forex': ('ForexLoader', 'finance', 'forecast'),

        # Healthcare
        'physionet_ecg': ('PhysioNetLoader', 'healthcare', 'classify'),
        'physionet_eeg': ('PhysioNetLoader', 'healthcare', 'detect'),
        'mimic_vitals': ('MIMICLoader', 'healthcare', 'forecast'),

        # Energy
        'electricity_load': ('UCILoader', 'energy', 'forecast'),
        'solar_power': ('SolarLoader', 'energy', 'forecast'),
        'building_energy': ('BuildingLoader', 'energy', 'forecast'),

        # IoT/Sensors
        'air_quality': ('UCILoader', 'iot', 'forecast'),
        'yahoo_anomaly': ('YahooLoader', 'iot', 'detect'),
        'numenta_anomaly': ('NumentaLoader', 'iot', 'detect'),
        'server_metrics': ('ServerLoader', 'iot', 'detect'),

        # Traffic
        'pems_traffic': ('PEMSLoader', 'traffic', 'forecast'),
        'metr_la': ('METRLoader', 'traffic', 'forecast'),

        # Climate
        'temperature': ('ClimateLoader', 'climate', 'forecast'),
        'weather': ('WeatherLoader', 'climate', 'forecast'),

        # UCR Archive (128 datasets)
        'ucr_*': ('UCRLoader', 'other', 'classify'),  # wildcard
    }

    def __init__(self, data_dir: str = './data/tsdb'):
        """
        Parameters
        ----------
        data_dir : str
            Root directory for downloaded datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize loaders
        self._loaders = {}

    def load(self, dataset_name: str, **kwargs) -> TimeSeriesData:
        """
        Load dataset by name

        Parameters
        ----------
        dataset_name : str
            Name of dataset (e.g., 'electricity_load', 'ucr_ECG200')
        **kwargs
            Additional loader-specific arguments

        Returns
        -------
        TimeSeriesData
            Loaded time-series data with metadata
        """
        loader_name, domain, task_type = self._get_loader_info(dataset_name)

        # Get or create loader
        if loader_name not in self._loaders:
            self._loaders[loader_name] = self._create_loader(loader_name)

        loader = self._loaders[loader_name]

        # Load data
        data = loader.load(dataset_name, **kwargs)

        # Ensure metadata is populated
        if data.metadata.domain == 'other':
            data.metadata.domain = domain
        if data.metadata.task_type is None:
            data.metadata.task_type = task_type

        return data

    def _get_loader_info(self, dataset_name: str) -> Tuple[str, str, str]:
        """Get loader class name, domain, and task type for dataset"""
        # Direct match
        if dataset_name in self.DATASET_REGISTRY:
            return self.DATASET_REGISTRY[dataset_name]

        # Wildcard match (e.g., ucr_*)
        for pattern, info in self.DATASET_REGISTRY.items():
            if '*' in pattern:
                prefix = pattern.split('*')[0]
                if dataset_name.startswith(prefix):
                    return info

        raise ValueError(f"Unknown dataset: {dataset_name}")

    def _create_loader(self, loader_name: str) -> BaseLoader:
        """Instantiate loader by name"""
        # Import loaders dynamically
        from .ucr_loader import UCRLoader
        from .monash_loader import MonashLoader
        from .uci_loader import UCILoader
        from .yahoo_loader import YahooLoader

        loaders_map = {
            'UCRLoader': UCRLoader,
            'MonashLoader': MonashLoader,
            'UCILoader': UCILoader,
            'YahooLoader': YahooLoader,
            # Add more as implemented
        }

        if loader_name not in loaders_map:
            # Default fallback
            return GenericLoader(self.data_dir)

        return loaders_map[loader_name](self.data_dir)

    def list_datasets(self) -> List[str]:
        """List all available datasets"""
        datasets = []
        for name in self.DATASET_REGISTRY.keys():
            if '*' not in name:
                datasets.append(name)
        return sorted(datasets)

    def get_domain_datasets(self, domain: str) -> List[str]:
        """Get all datasets for a specific domain"""
        return [
            name for name, (_, dom, _) in self.DATASET_REGISTRY.items()
            if dom == domain and '*' not in name
        ]


class GenericLoader(BaseLoader):
    """
    Generic fallback loader for simple CSV/numpy formats
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def load(self, dataset_name: str, **kwargs) -> TimeSeriesData:
        """
        Load from standard format:
        - {dataset_name}_train.npy, {dataset_name}_test.npy
        or
        - {dataset_name}.csv with train/test split
        """
        dataset_path = self.data_dir / dataset_name

        # Try .npy format
        train_path = dataset_path / f"{dataset_name}_train.npy"
        test_path = dataset_path / f"{dataset_name}_test.npy"

        if train_path.exists() and test_path.exists():
            X_train = np.load(train_path)
            X_test = np.load(test_path)

            # Assume last column is target for forecasting
            if X_train.ndim == 2:
                X_train = X_train.reshape(X_train.shape[0], -1, 1)
            if X_test.ndim == 2:
                X_test = X_test.reshape(X_test.shape[0], -1, 1)

            # Dummy targets (last timestep)
            y_train = X_train[:, -1, 0]
            y_test = X_test[:, -1, 0]

        else:
            raise FileNotFoundError(f"Dataset not found: {dataset_name}")

        # Create metadata
        metadata = DatasetMetadata(
            name=dataset_name,
            domain='other',
            task_type='forecast',
            n_series=len(X_train),
            seq_len=X_train.shape[1],
            n_features=X_train.shape[2],
            frequency='unknown',
            has_missing=np.isnan(X_train).any(),
            train_size=len(X_train),
            test_size=len(X_test)
        )

        return TimeSeriesData(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            metadata=metadata
        )

    def list_datasets(self) -> List[str]:
        """List datasets in data directory"""
        if not self.data_dir.exists():
            return []
        return [d.name for d in self.data_dir.iterdir() if d.is_dir()]


def preprocess_time_series(
    X: np.ndarray,
    method: str = 'standardize',
    axis: int = 0
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Preprocess time series data

    Parameters
    ----------
    X : np.ndarray
        Time series data
    method : str
        'standardize', 'normalize', or 'none'
    axis : int
        Axis along which to compute statistics

    Returns
    -------
    X_processed : np.ndarray
        Processed data
    stats : dict
        Statistics for inverse transform
    """
    if method == 'standardize':
        mean = np.nanmean(X, axis=axis, keepdims=True)
        std = np.nanstd(X, axis=axis, keepdims=True) + 1e-8
        X_processed = (X - mean) / std
        stats = {'mean': mean, 'std': std}

    elif method == 'normalize':
        min_val = np.nanmin(X, axis=axis, keepdims=True)
        max_val = np.nanmax(X, axis=axis, keepdims=True)
        X_processed = (X - min_val) / (max_val - min_val + 1e-8)
        stats = {'min': min_val, 'max': max_val}

    else:  # none
        X_processed = X
        stats = {}

    return X_processed, stats
