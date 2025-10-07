#!/usr/bin/env python3
"""
Monash Time Series Forecasting Repository Loader

30+ forecasting datasets across multiple domains
Source: https://forecastingdata.org/
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from .tsdb_loader import BaseLoader, TimeSeriesData, DatasetMetadata


class MonashLoader(BaseLoader):
    """
    Loader for Monash Forecasting Repository

    Datasets include: tourism, electricity, traffic, solar, weather, etc.
    """

    DATASETS = {
        'tourism_monthly': {
            'domain': 'other',
            'frequency': 'monthly',
            'seasonality': 12,
            'horizon': 24
        },
        'tourism_quarterly': {
            'domain': 'other',
            'frequency': 'quarterly',
            'seasonality': 4,
            'horizon': 8
        },
        'tourism_yearly': {
            'domain': 'other',
            'frequency': 'yearly',
            'seasonality': 1,
            'horizon': 4
        },
        'cif_2016': {
            'domain': 'finance',
            'frequency': 'monthly',
            'seasonality': 12,
            'horizon': 12
        },
        'london_smart_meters': {
            'domain': 'energy',
            'frequency': 'half_hourly',
            'seasonality': 48,
            'horizon': 48
        },
        'australian_electricity': {
            'domain': 'energy',
            'frequency': 'half_hourly',
            'seasonality': 48,
            'horizon': 48
        },
        'wind_farms': {
            'domain': 'energy',
            'frequency': 'hourly',
            'seasonality': 24,
            'horizon': 24
        },
        'solar_power': {
            'domain': 'energy',
            'frequency': 'hourly',
            'seasonality': 24,
            'horizon': 24
        },
        'pedestrian_counts': {
            'domain': 'traffic',
            'frequency': 'hourly',
            'seasonality': 24,
            'horizon': 24
        },
        'traffic': {
            'domain': 'traffic',
            'frequency': 'hourly',
            'seasonality': 24,
            'horizon': 24
        },
        'electricity_weekly': {
            'domain': 'energy',
            'frequency': 'weekly',
            'seasonality': 52,
            'horizon': 8
        },
        'm4_hourly': {
            'domain': 'other',
            'frequency': 'hourly',
            'seasonality': 24,
            'horizon': 48
        },
        'm4_daily': {
            'domain': 'other',
            'frequency': 'daily',
            'seasonality': 7,
            'horizon': 14
        },
        'm4_weekly': {
            'domain': 'other',
            'frequency': 'weekly',
            'seasonality': 52,
            'horizon': 13
        },
        'm4_monthly': {
            'domain': 'other',
            'frequency': 'monthly',
            'seasonality': 12,
            'horizon': 18
        },
        'm4_quarterly': {
            'domain': 'other',
            'frequency': 'quarterly',
            'seasonality': 4,
            'horizon': 8
        },
        'm4_yearly': {
            'domain': 'other',
            'frequency': 'yearly',
            'seasonality': 1,
            'horizon': 6
        },
        'nn5_daily': {
            'domain': 'finance',
            'frequency': 'daily',
            'seasonality': 7,
            'horizon': 56
        },
        'kaggle_web_traffic': {
            'domain': 'other',
            'frequency': 'daily',
            'seasonality': 7,
            'horizon': 59
        },
        'weather': {
            'domain': 'climate',
            'frequency': 'daily',
            'seasonality': 365,
            'horizon': 30
        },
        'sunspot': {
            'domain': 'climate',
            'frequency': 'monthly',
            'seasonality': 132,
            'horizon': 12
        }
    }

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir) / 'monash'
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        dataset_name: str,
        horizon: Optional[int] = None,
        train_ratio: float = 0.8,
        **kwargs
    ) -> TimeSeriesData:
        """
        Load Monash dataset

        Parameters
        ----------
        dataset_name : str
            Dataset name from Monash repository
        horizon : int, optional
            Forecasting horizon (default from dataset config)
        train_ratio : float
            Train/test split ratio

        Returns
        -------
        TimeSeriesData
        """
        if dataset_name not in self.DATASETS:
            raise ValueError(
                f"Unknown Monash dataset: {dataset_name}. "
                f"Available: {list(self.DATASETS.keys())}"
            )

        config = self.DATASETS[dataset_name]
        if horizon is None:
            horizon = config['horizon']

        # Load dataset file
        dataset_path = self.data_dir / f"{dataset_name}.csv"

        if not dataset_path.exists():
            print(f"[MonashLoader] Dataset '{dataset_name}' not found.")
            print(f"Please download from: https://forecastingdata.org/")
            print(f"Save to: {dataset_path}")
            # Create dummy data for testing
            return self._create_dummy_data(dataset_name, config, horizon)

        # Load CSV
        df = pd.read_csv(dataset_path)

        # Convert to time series format
        # Assume columns: series_id, timestamp, value
        X_all = []
        y_all = []

        series_ids = df['series_id'].unique() if 'series_id' in df.columns else [0]

        for sid in series_ids:
            if 'series_id' in df.columns:
                series_data = df[df['series_id'] == sid]['value'].values
            else:
                series_data = df['value'].values

            # Create sliding windows
            seq_len = 100  # default lookback
            for i in range(len(series_data) - seq_len - horizon + 1):
                X_all.append(series_data[i:i+seq_len])
                y_all.append(series_data[i+seq_len:i+seq_len+horizon])

        X_all = np.array(X_all).reshape(-1, seq_len, 1)
        y_all = np.array(y_all)

        # Train/test split
        n_train = int(len(X_all) * train_ratio)
        X_train = X_all[:n_train]
        y_train = y_all[:n_train]
        X_test = X_all[n_train:]
        y_test = y_all[n_train:]

        # Metadata
        metadata = DatasetMetadata(
            name=dataset_name,
            domain=config['domain'],
            task_type='forecast',
            n_series=len(series_ids),
            seq_len=seq_len,
            n_features=1,
            frequency=config['frequency'],
            has_missing=False,
            seasonality=config['seasonality'],
            horizon=horizon,
            train_size=n_train,
            test_size=len(X_test)
        )

        return TimeSeriesData(
            X_train=X_train.astype(np.float32),
            y_train=y_train.astype(np.float32),
            X_test=X_test.astype(np.float32),
            y_test=y_test.astype(np.float32),
            metadata=metadata
        )

    def list_datasets(self) -> List[str]:
        """List available Monash datasets"""
        return list(self.DATASETS.keys())

    def _create_dummy_data(
        self,
        dataset_name: str,
        config: dict,
        horizon: int
    ) -> TimeSeriesData:
        """Create dummy synthetic data for testing"""
        n_train = 1000
        n_test = 200
        seq_len = 100

        # Generate synthetic time series with seasonality
        seasonality = config['seasonality']
        t_train = np.arange(n_train * (seq_len + horizon))
        trend = 0.01 * t_train
        seasonal = 10 * np.sin(2 * np.pi * t_train / seasonality)
        noise = np.random.randn(len(t_train))
        series = trend + seasonal + noise

        # Create windows
        X_train = []
        y_train = []
        for i in range(n_train):
            X_train.append(series[i:i+seq_len])
            y_train.append(series[i+seq_len:i+seq_len+horizon])

        X_train = np.array(X_train).reshape(-1, seq_len, 1)
        y_train = np.array(y_train)

        # Test set
        t_test = np.arange(n_test * (seq_len + horizon)) + len(t_train)
        trend_test = 0.01 * t_test
        seasonal_test = 10 * np.sin(2 * np.pi * t_test / seasonality)
        noise_test = np.random.randn(len(t_test))
        series_test = trend_test + seasonal_test + noise_test

        X_test = []
        y_test = []
        for i in range(n_test):
            X_test.append(series_test[i:i+seq_len])
            y_test.append(series_test[i+seq_len:i+seq_len+horizon])

        X_test = np.array(X_test).reshape(-1, seq_len, 1)
        y_test = np.array(y_test)

        metadata = DatasetMetadata(
            name=dataset_name + '_dummy',
            domain=config['domain'],
            task_type='forecast',
            n_series=1,
            seq_len=seq_len,
            n_features=1,
            frequency=config['frequency'],
            has_missing=False,
            seasonality=seasonality,
            horizon=horizon,
            train_size=n_train,
            test_size=n_test
        )

        return TimeSeriesData(
            X_train=X_train.astype(np.float32),
            y_train=y_train.astype(np.float32),
            X_test=X_test.astype(np.float32),
            y_test=y_test.astype(np.float32),
            metadata=metadata
        )
