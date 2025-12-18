"""
Utility module containing boundary handling, termination criteria, and adaptive mechanisms.
"""

from pyrade.utils.boundary import (
    BoundaryHandler,
    ClipBoundary,
    ReflectBoundary,
    RandomBoundary,
    WrapBoundary,
    MidpointBoundary,
)
from pyrade.utils.termination import (
    TerminationCriterion,
    MaxIterations,
    FitnessThreshold,
    NoImprovement,
    MaxTime,
    FitnessVariance,
)
from pyrade.utils.adaptation import (
    AdaptivePopulationSize,
    ParameterEnsemble,
)

__all__ = [
    "BoundaryHandler",
    "ClipBoundary",
    "ReflectBoundary",
    "RandomBoundary",
    "WrapBoundary",
    "MidpointBoundary",
    "TerminationCriterion",
    "MaxIterations",
    "FitnessThreshold",
    "NoImprovement",
    "MaxTime",
    "FitnessVariance",
    "AdaptivePopulationSize",
    "ParameterEnsemble",
]
