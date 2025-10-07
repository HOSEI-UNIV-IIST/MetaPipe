#!/usr/bin/env python3
"""UCI ML Repository Time-Series Loader (Air Quality, Electricity, etc.)"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import List
from .tsdb_loader import BaseLoader, TimeSeriesData, DatasetMetadata


class UCILoader(BaseLoader):
    """Loader for UCI time-series datasets"""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir) / 'uci'
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load(self, dataset_name: str, **kwargs) -> TimeSeriesData:
        """Load UCI dataset (stub - implement specific loaders)"""
        raise NotImplementedError(f"UCI loader for {dataset_name} not yet implemented")

    def list_datasets(self) -> List[str]:
        return ['air_quality', 'electricity_load', 'building_energy']
