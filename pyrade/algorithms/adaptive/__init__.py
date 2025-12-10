"""
Adaptive Differential Evolution Variants

This module contains state-of-the-art adaptive DE algorithms that automatically
adjust their parameters during optimization.
"""

from .jde import jDE
from .sade import SaDE
from .jade import JADE
from .shade import SHADE
from .lshade import LSHADE
from .lshade_epsin import LSHADEEpSin
from .jso import jSO
from .apsde import APSDE

__all__ = [
    'jDE',
    'SaDE',
    'JADE',
    'SHADE',
    'LSHADE',
    'LSHADEEpSin',
    'jSO',
    'APSDE',
]
