"""
PyRADE - Python Rapid Algorithm for Differential Evolution

A high-performance, modular Differential Evolution optimization package
with clean OOP architecture and aggressive vectorization.

Example usage:
-------------
>>> from pyrade import DifferentialEvolution
>>> from pyrade.operators import DErand1, BinomialCrossover
>>> 
>>> def sphere(x):
...     return sum(x**2)
>>> 
>>> optimizer = DifferentialEvolution(
...     objective_func=sphere,
...     bounds=(-100, 100),
...     mutation=DErand1(F=0.8),
...     crossover=BinomialCrossover(CR=0.9),
...     pop_size=50,
...     max_iter=1000
... )
>>> 
>>> result = optimizer.optimize()
>>> print(f"Best solution: {result['best_solution']}")
>>> print(f"Best fitness: {result['best_fitness']}")
"""

from pyrade.__version__ import __version__

__author__ = "PyRADE Contributors"

from pyrade.core.algorithm import DifferentialEvolution
from pyrade.core.population import Population

__all__ = [
    "DifferentialEvolution",
    "Population",
]
