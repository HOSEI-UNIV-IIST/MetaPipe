"""Policy modules for MetaPipe"""

from .map import MAPPolicy, QNetwork, HorizonAggregator, HorizonReward
from .map_optimized import (
    FastMAPPolicy,
    FastQNetwork
)

__all__ = [
    "MAPPolicy",
    "QNetwork",
    "HorizonAggregator",
    "HorizonReward",
    "FastMAPPolicy",
    "FastQNetwork"
]
