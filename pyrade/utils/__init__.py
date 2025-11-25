"""
Utility module containing boundary handling and termination criteria.
"""

from pyrade.utils.boundary import (
    BoundaryHandler,
    ClipBoundary,
    ReflectBoundary,
    RandomBoundary,
)
from pyrade.utils.termination import (
    TerminationCriterion,
    MaxIterations,
    FitnessThreshold,
    NoImprovement,
)

__all__ = [
    "BoundaryHandler",
    "ClipBoundary",
    "ReflectBoundary",
    "RandomBoundary",
    "TerminationCriterion",
    "MaxIterations",
    "FitnessThreshold",
    "NoImprovement",
]
