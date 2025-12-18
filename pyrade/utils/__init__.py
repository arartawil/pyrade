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
from pyrade.utils.opposition import (
    OppositionBasedLearning,
    apply_obl_initialization,
)
from pyrade.utils.chaotic import (
    ChaoticMap,
    LogisticMap,
    TentMap,
    SineMap,
    ChebyshevMap,
    ChaoticParameterController,
    create_chaotic_controller,
)

__all__ = [
    # Boundary handling
    "BoundaryHandler",
    "ClipBoundary",
    "ReflectBoundary",
    "RandomBoundary",
    "WrapBoundary",
    "MidpointBoundary",
    # Termination criteria
    "TerminationCriterion",
    "MaxIterations",
    "FitnessThreshold",
    "NoImprovement",
    "MaxTime",
    "FitnessVariance",
    # Adaptive mechanisms (v0.4.2)
    "AdaptivePopulationSize",
    "ParameterEnsemble",
    # Opposition-based learning (v0.4.4)
    "OppositionBasedLearning",
    "apply_obl_initialization",
    # Chaotic maps (v0.4.4)
    "ChaoticMap",
    "LogisticMap",
    "TentMap",
    "SineMap",
    "ChebyshevMap",
    "ChaoticParameterController",
    "create_chaotic_controller",
]
