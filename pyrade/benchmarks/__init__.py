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
# TODO: Fix CEC2017Function - requires core.problem.Problem base class
# from .cec2017 import CEC2017Function

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
