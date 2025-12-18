"""
Opposition-Based Learning (OBL) for Differential Evolution.

Opposition-based learning is a machine intelligence strategy that considers
both an estimate and its opposite to find a better approximation of the solution.
This module provides OBL initialization and generation jumping for DE.
"""

import numpy as np
from typing import Tuple, Callable, Optional


class OppositionBasedLearning:
    """
    Opposition-Based Learning for population initialization and generation jumping.
    
    OBL is based on the concept that the opposite of a candidate solution may be
    closer to the global optimum than the candidate itself. By simultaneously
    evaluating a candidate and its opposite, we increase the chance of finding
    better solutions.
    
    Types of Opposition:
    - **Simple Opposition**: opposite = lower + upper - solution
    - **Quasi Opposition**: opposite = (lower + upper)/2 + rand * ((lower + upper)/2 - solution)
    - **Quasi-Reflected Opposition**: Combines reflection with randomization
    - **Generalized Opposition**: opposite = lower + upper - k * solution (k is random)
    
    Parameters
    ----------
    opposition_type : str, default='simple'
        Type of opposition: 'simple', 'quasi', 'quasi-reflected', 'generalized'
    jumping_rate : float, default=0.3
        Probability of applying generation jumping (0-1)
    
    Attributes
    ----------
    opposition_type : str
        The type of opposition being used
    jumping_rate : float
        Generation jumping probability
    
    References
    ----------
    Tizhoosh, H. R. (2005). Opposition-based learning: a new scheme for machine
    intelligence. In International conference on computational intelligence for
    modelling, control and automation.
    
    Examples
    --------
    >>> obl = OppositionBasedLearning(opposition_type='simple')
    >>> # Initialize population with opposition
    >>> pop, fitness = obl.initialize_population(
    ...     objective_func=sphere,
    ...     pop_size=50,
    ...     bounds=(-100, 100),
    ...     dimensions=30
    ... )
    >>> # Apply generation jumping
    >>> pop, fitness = obl.generation_jumping(
    ...     population=pop,
    ...     fitness=fitness,
    ...     objective_func=sphere,
    ...     bounds=(-100, 100)
    ... )
    """
    
    def __init__(
        self,
        opposition_type: str = 'simple',
        jumping_rate: float = 0.3
    ):
        valid_types = ['simple', 'quasi', 'quasi-reflected', 'generalized']
        if opposition_type not in valid_types:
            raise ValueError(f"opposition_type must be one of {valid_types}")
        if not 0 <= jumping_rate <= 1:
            raise ValueError("jumping_rate must be in [0, 1]")
        
        self.opposition_type = opposition_type
        self.jumping_rate = jumping_rate
    
    def compute_opposite(
        self,
        solution: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray
    ) -> np.ndarray:
        """
        Compute the opposite solution based on the opposition type.
        
        Parameters
        ----------
        solution : ndarray, shape (dim,) or (pop_size, dim)
            Original solution(s)
        lower_bounds : ndarray, shape (dim,)
            Lower bounds for each dimension
        upper_bounds : ndarray, shape (dim,)
            Upper bounds for each dimension
            
        Returns
        -------
        opposite : ndarray
            Opposite solution(s) with same shape as input
        """
        if self.opposition_type == 'simple':
            # O_x = a + b - x
            opposite = lower_bounds + upper_bounds - solution
            
        elif self.opposition_type == 'quasi':
            # O_x = c + rand * (c - x), where c = (a+b)/2
            center = (lower_bounds + upper_bounds) / 2
            if solution.ndim == 1:
                rand_factor = np.random.rand(len(solution))
            else:
                rand_factor = np.random.rand(*solution.shape)
            opposite = center + rand_factor * (center - solution)
            
        elif self.opposition_type == 'quasi-reflected':
            # Combination of reflection and randomization
            center = (lower_bounds + upper_bounds) / 2
            if solution.ndim == 1:
                rand_factor = np.random.uniform(0.5, 1.5, len(solution))
            else:
                rand_factor = np.random.uniform(0.5, 1.5, solution.shape)
            opposite = center + rand_factor * (center - solution)
            
        elif self.opposition_type == 'generalized':
            # O_x = a + b - k*x, where k is random
            if solution.ndim == 1:
                k = np.random.uniform(0.5, 1.5, len(solution))
            else:
                k = np.random.uniform(0.5, 1.5, solution.shape)
            opposite = lower_bounds + upper_bounds - k * solution
        
        # Ensure opposite is within bounds
        opposite = np.clip(opposite, lower_bounds, upper_bounds)
        
        return opposite
    
    def initialize_population(
        self,
        objective_func: Callable,
        pop_size: int,
        bounds: Tuple[float, float],
        dimensions: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize population using opposition-based learning.
        
        Creates pop_size random solutions and their opposites, then selects
        the best pop_size individuals from the combined set.
        
        Parameters
        ----------
        objective_func : callable
            Objective function to minimize: f(x) -> float
        pop_size : int
            Population size
        bounds : tuple
            (lower_bound, upper_bound)
        dimensions : int
            Problem dimensionality
            
        Returns
        -------
        population : ndarray, shape (pop_size, dimensions)
            Initial population
        fitness : ndarray, shape (pop_size,)
            Fitness values for population
        """
        lower, upper = bounds
        lower_bounds = np.full(dimensions, lower)
        upper_bounds = np.full(dimensions, upper)
        
        # Generate random population
        population = np.random.uniform(lower, upper, (pop_size, dimensions))
        
        # Compute opposite population
        opposite_population = self.compute_opposite(
            population, lower_bounds, upper_bounds
        )
        
        # Combine and evaluate all solutions
        combined_pop = np.vstack([population, opposite_population])
        combined_fitness = np.array([objective_func(ind) for ind in combined_pop])
        
        # Select best pop_size individuals
        best_indices = np.argsort(combined_fitness)[:pop_size]
        final_population = combined_pop[best_indices]
        final_fitness = combined_fitness[best_indices]
        
        return final_population, final_fitness
    
    def generation_jumping(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        objective_func: Callable,
        bounds: Tuple[float, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply generation jumping using opposition-based learning.
        
        With probability jumping_rate, computes opposite population and
        replaces worse individuals with better opposite individuals.
        
        Parameters
        ----------
        population : ndarray, shape (pop_size, dim)
            Current population
        fitness : ndarray, shape (pop_size,)
            Current fitness values
        objective_func : callable
            Objective function to minimize
        bounds : tuple
            (lower_bound, upper_bound)
            
        Returns
        -------
        population : ndarray
            Updated population
        fitness : ndarray
            Updated fitness values
        """
        # Apply jumping with probability
        if np.random.rand() > self.jumping_rate:
            return population, fitness
        
        pop_size, dim = population.shape
        lower, upper = bounds
        lower_bounds = np.full(dim, lower)
        upper_bounds = np.full(dim, upper)
        
        # Compute opposite population
        opposite_pop = self.compute_opposite(
            population, lower_bounds, upper_bounds
        )
        
        # Evaluate opposite population
        opposite_fitness = np.array([objective_func(ind) for ind in opposite_pop])
        
        # Replace worse individuals with better opposites
        improved_mask = opposite_fitness < fitness
        population[improved_mask] = opposite_pop[improved_mask]
        fitness[improved_mask] = opposite_fitness[improved_mask]
        
        return population, fitness
    
    def get_opposition_info(self) -> dict:
        """
        Get information about the opposition strategy.
        
        Returns
        -------
        info : dict
            Dictionary with opposition strategy details
        """
        return {
            'opposition_type': self.opposition_type,
            'jumping_rate': self.jumping_rate,
            'description': self._get_description()
        }
    
    def _get_description(self) -> str:
        """Get description of the opposition type."""
        descriptions = {
            'simple': 'Simple opposition: O_x = a + b - x',
            'quasi': 'Quasi opposition: O_x = c + rand * (c - x)',
            'quasi-reflected': 'Quasi-reflected opposition with randomization',
            'generalized': 'Generalized opposition: O_x = a + b - k*x'
        }
        return descriptions.get(self.opposition_type, 'Unknown')


def apply_obl_initialization(
    objective_func: Callable,
    pop_size: int,
    bounds: Tuple[float, float],
    dimensions: int,
    opposition_type: str = 'simple'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for OBL initialization.
    
    Parameters
    ----------
    objective_func : callable
        Objective function to minimize
    pop_size : int
        Population size
    bounds : tuple
        (lower_bound, upper_bound)
    dimensions : int
        Problem dimensionality
    opposition_type : str, default='simple'
        Type of opposition
        
    Returns
    -------
    population : ndarray
        Initial population
    fitness : ndarray
        Fitness values
        
    Examples
    --------
    >>> pop, fit = apply_obl_initialization(
    ...     objective_func=sphere,
    ...     pop_size=50,
    ...     bounds=(-100, 100),
    ...     dimensions=30
    ... )
    """
    obl = OppositionBasedLearning(opposition_type=opposition_type)
    return obl.initialize_population(objective_func, pop_size, bounds, dimensions)
