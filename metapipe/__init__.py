"""
MetaPipe: Temporal Context-Aware Routing for Multi-Stage Time-Series LLM Pipelines

A novel framework for adaptive routing in time-series analysis pipelines with:
- TCAR: Temporal Context-Aware Routing
- MAP: Multi-Horizon Adaptive Policy
- BCPR: Budget-Constrained Pareto Routing
- UQE: Uncertainty-Quantified Escalation
- CPTL: Cross-Pipeline Transfer Learning
"""

__version__ = "0.1.0"
__author__ = "Franck J. A. Messou"

from .features.tcar import TCARExtractor
from .policy.map import MAPPolicy
from .optimizer.bcpr import BCPROptimizer
from .uncertainty.uqe import ConformalEscalation
from .transfer.cptl import MetaRouter

__all__ = [
    "TCARExtractor",
    "MAPPolicy",
    "BCPROptimizer",
    "ConformalEscalation",
    "MetaRouter",
]
