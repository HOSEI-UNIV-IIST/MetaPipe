#!/usr/bin/env python3
"""Yahoo/Numenta Anomaly Detection Benchmark Loader"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import List
from .tsdb_loader import BaseLoader, TimeSeriesData, DatasetMetadata


class YahooLoader(BaseLoader):
    """Loader for Yahoo anomaly detection datasets"""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir) / 'yahoo'
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load(self, dataset_name: str, **kwargs) -> TimeSeriesData:
        """Load Yahoo anomaly dataset (stub)"""
        raise NotImplementedError(f"Yahoo loader for {dataset_name} not yet implemented")

    def list_datasets(self) -> List[str]:
        return ['yahoo_anomaly', 'numenta_anomaly']
