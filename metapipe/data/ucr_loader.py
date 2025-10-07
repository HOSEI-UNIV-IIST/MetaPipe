#!/usr/bin/env python3
"""
UCR Time Series Archive Loader

128+ univariate time-series classification datasets
Source: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from .tsdb_loader import BaseLoader, TimeSeriesData, DatasetMetadata


class UCRLoader(BaseLoader):
    """
    Loader for UCR Time Series Archive

    Datasets include: ECG200, GunPoint, ItalyPowerDemand, etc.
    """

    # Popular UCR datasets
    POPULAR_DATASETS = [
        'ECG200', 'ECG5000', 'ElectricDevices', 'FordA', 'FordB',
        'GunPoint', 'ItalyPowerDemand', 'MoteStrain', 'SonyAIBORobotSurface1',
        'TwoLeadECG', 'Wafer', 'Yoga', 'ACSF1', 'Adiac', 'ArrowHead',
        'Beef', 'BeetleFly', 'BirdChicken', 'CBF', 'ChlorineConcentration',
        'CinCECGTorso', 'Coffee', 'Computers', 'CricketX', 'CricketY',
        'CricketZ', 'DiatomSizeReduction', 'DistalPhalanxOutlineCorrect',
        'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxTW', 'Earthquakes',
        'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords',
        'Fish', 'FreezerRegularTrain', 'FreezerSmallTrain', 'Haptics',
        'Herring', 'InlineSkate', 'InsectWingbeatSound', 'LargeKitchenAppliances',
        'Lightning2', 'Lightning7', 'Mallat', 'Meat', 'MedicalImages',
        'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxOutlineAgeGroup',
        'MiddlePhalanxTW', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2',
        'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme',
        'Plane', 'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxOutlineAgeGroup',
        'ProximalPhalanxTW', 'RefrigerationDevices', 'ScreenType',
        'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'Strawberry',
        'SwedishLeaf', 'Symbols', 'SyntheticControl', 'ToeSegmentation1',
        'ToeSegmentation2', 'Trace', 'TwoPatterns', 'UWaveGestureLibraryAll',
        'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ',
        'WordSynonyms', 'Worms', 'WormsTwoClass'
    ]

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir) / 'ucr'
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load(self, dataset_name: str, **kwargs) -> TimeSeriesData:
        """
        Load UCR dataset

        Parameters
        ----------
        dataset_name : str
            Dataset name (e.g., 'ucr_ECG200' or 'ECG200')

        Returns
        -------
        TimeSeriesData
        """
        # Remove 'ucr_' prefix if present
        if dataset_name.startswith('ucr_'):
            dataset_name = dataset_name[4:]

        # Check if dataset exists locally
        dataset_path = self.data_dir / dataset_name
        if not dataset_path.exists():
            # Try to download
            self._download_dataset(dataset_name)

        # Load train and test files
        train_file = dataset_path / f"{dataset_name}_TRAIN.tsv"
        test_file = dataset_path / f"{dataset_name}_TEST.tsv"

        if not train_file.exists():
            # Try alternative naming
            train_file = dataset_path / f"{dataset_name}_TRAIN.txt"
            test_file = dataset_path / f"{dataset_name}_TEST.txt"

        if not train_file.exists():
            raise FileNotFoundError(
                f"UCR dataset not found: {dataset_name}. "
                f"Please download from: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/"
            )

        # Load data (format: label, value1, value2, ...)
        train_data = pd.read_csv(train_file, sep='\\s+', header=None).values
        test_data = pd.read_csv(test_file, sep='\\s+', header=None).values

        # Split into X and y
        y_train = train_data[:, 0].astype(int)
        X_train = train_data[:, 1:]

        y_test = test_data[:, 0].astype(int)
        X_test = test_data[:, 1:]

        # Reshape to (n_samples, seq_len, 1) for univariate
        X_train = X_train.reshape(X_train.shape[0], -1, 1)
        X_test = X_test.reshape(X_test.shape[0], -1, 1)

        # Create metadata
        metadata = DatasetMetadata(
            name=dataset_name,
            domain='other',
            task_type='classify',
            n_series=len(X_train) + len(X_test),
            seq_len=X_train.shape[1],
            n_features=1,
            frequency='unknown',
            has_missing=False,
            train_size=len(X_train),
            test_size=len(X_test)
        )

        return TimeSeriesData(
            X_train=X_train.astype(np.float32),
            y_train=y_train,
            X_test=X_test.astype(np.float32),
            y_test=y_test,
            metadata=metadata
        )

    def list_datasets(self) -> List[str]:
        """List available UCR datasets"""
        return self.POPULAR_DATASETS

    def _download_dataset(self, dataset_name: str):
        """
        Download UCR dataset (placeholder - manual download required)
        """
        print(f"[UCRLoader] Dataset '{dataset_name}' not found locally.")
        print(f"Please download from: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/")
        print(f"Extract to: {self.data_dir / dataset_name}")
