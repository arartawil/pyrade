"""
Benchmark functions module for testing optimization algorithms.
"""

from pyrade.benchmarks.functions import (
    BenchmarkFunction,
    Sphere,
    Rastrigin,
    Rosenbrock,
    Ackley,
    Griewank,
    Schwefel,
    Levy,
    Michalewicz,
    Zakharov,
    Easom,
    StyblinskiTang,
)
from .cec2017 import CEC2017Function

__all__ = [
    "BenchmarkFunction",
    "Sphere",
    "Rastrigin",
    "Rosenbrock",
    "Ackley",
    "Griewank",
    "Schwefel",
    "Levy",
    "Michalewicz",
    "Zakharov",
    "Easom",
    "StyblinskiTang",
]

__all__.append("CEC2017Function")
