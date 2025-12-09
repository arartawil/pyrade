"""
Population management for Differential Evolution.

This module provides efficient vectorized population operations.
"""

import numpy as np


class Population:
    """
    Manages the population of candidate solutions.
    
    This class handles population initialization, storage, and
    provides vectorized operations for efficient population management.
    
    Parameters
    ----------
    pop_size : int
        Size of the population
    dim : int
        Dimensionality of the search space
    bounds : tuple or array-like
        Search space bounds (lb, ub) or [(lb1, ub1), ...]
    seed : int, optional
        Random seed for reproducibility
    
    Attributes
    ----------
    vectors : ndarray, shape (pop_size, dim)
        Population vectors
    fitness : ndarray, shape (pop_size,)
        Fitness values for each individual
    best_idx : int
        Index of the best individual
    best_vector : ndarray, shape (dim,)
        Best solution vector
    best_fitness : float
        Best fitness value
    """
    
    def __init__(self, pop_size, dim, bounds, seed=None):
        """Initialize population."""
        self.pop_size = pop_size
        self.dim = dim
        self.seed = seed
        
        # Parse bounds
        self.lb, self.ub = self._parse_bounds(bounds, dim)
        
        # Initialize arrays
        self.vectors = np.zeros((pop_size, dim))
        self.fitness = np.full(pop_size, np.inf)
        self.best_idx = 0
        self.best_vector = None
        self.best_fitness = np.inf
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
    
    def _parse_bounds(self, bounds, dim):
        """
        Parse bounds into lower and upper bound arrays.
        
        Parameters
        ----------
        bounds : tuple or array-like
            Either (lb, ub) or [(lb1, ub1), (lb2, ub2), ...]
        dim : int
            Dimensionality
        
        Returns
        -------
        lb : ndarray, shape (dim,)
            Lower bounds
        ub : ndarray, shape (dim,)
            Upper bounds
        """
        bounds = np.array(bounds)
        
        if bounds.ndim == 1 and len(bounds) == 2:
            # Uniform bounds (lb, ub)
            lb = np.full(dim, bounds[0])
            ub = np.full(dim, bounds[1])
        elif bounds.ndim == 2 and bounds.shape[0] == dim and bounds.shape[1] == 2:
            # Per-dimension bounds [(lb1, ub1), ...]
            lb = bounds[:, 0]
            ub = bounds[:, 1]
        else:
            raise ValueError(
                f"Invalid bounds shape. Expected (2,) or ({dim}, 2), got {bounds.shape}"
            )
        
        if np.any(lb >= ub):
            raise ValueError("Lower bounds must be less than upper bounds")
        
        return lb, ub
    
    def initialize_random(self):
        """
        Initialize population with random vectors within bounds.
        
        Uses Latin Hypercube Sampling for better space coverage.
        """
        # Latin Hypercube Sampling for better initial distribution
        lhs_samples = np.zeros((self.pop_size, self.dim))
        
        for i in range(self.dim):
            # Divide range into pop_size intervals
            intervals = np.linspace(0, 1, self.pop_size + 1)
            # Sample randomly within each interval
            samples = np.random.uniform(intervals[:-1], intervals[1:])
            # Shuffle to avoid correlation
            np.random.shuffle(samples)
            lhs_samples[:, i] = samples
        
        # Scale to actual bounds
        self.vectors = self.lb + lhs_samples * (self.ub - self.lb)
    
    def evaluate(self, objective_func):
        """
        Evaluate fitness for all individuals in population.
        
        Parameters
        ----------
        objective_func : callable
            Objective function to minimize
        
        Returns
        -------
        fitness : ndarray, shape (pop_size,)
            Fitness values
        """
        # Memory-efficient evaluation with cleanup
        for i in range(self.pop_size):
            try:
                fitness_val = objective_func(self.vectors[i])
                # Handle inf/nan from objective function
                if not np.isfinite(fitness_val):
                    fitness_val = 1e100  # Large penalty for invalid values
                self.fitness[i] = fitness_val
            except Exception:
                # If evaluation fails, assign large penalty
                self.fitness[i] = 1e100
        
        self._update_best()
        return self.fitness
    
    def evaluate_vectors(self, vectors, objective_func):
        """
        Evaluate fitness for given vectors.
        
        Parameters
        ----------
        vectors : ndarray, shape (pop_size, dim)
            Vectors to evaluate
        objective_func : callable
            Objective function to minimize
        
        Returns
        -------
        fitness : ndarray, shape (pop_size,)
            Fitness values
        """
        # Memory-efficient evaluation with proper cleanup
        fitness = np.zeros(len(vectors), dtype=np.float64)
        for i, vec in enumerate(vectors):
            try:
                fitness_val = objective_func(vec)
                # Handle inf/nan from objective function
                if not np.isfinite(fitness_val):
                    fitness_val = 1e100
                fitness[i] = fitness_val
            except Exception:
                fitness[i] = 1e100
        return fitness
    
    def _update_best(self):
        """Update best solution information."""
        self.best_idx = np.argmin(self.fitness)
        self.best_vector = self.vectors[self.best_idx].copy()
        self.best_fitness = self.fitness[self.best_idx]
    
    def update(self, new_vectors, new_fitness):
        """
        Update population with new vectors and fitness.
        
        Parameters
        ----------
        new_vectors : ndarray, shape (pop_size, dim)
            New population vectors
        new_fitness : ndarray, shape (pop_size,)
            New fitness values
        """
        self.vectors = new_vectors.copy()
        self.fitness = new_fitness.copy()
        self._update_best()
    
    def get_indices(self):
        """Get array of population indices."""
        return np.arange(self.pop_size)
    
    def clip_to_bounds(self, vectors):
        """
        Clip vectors to bounds.
        
        Parameters
        ----------
        vectors : ndarray, shape (pop_size, dim)
            Vectors to clip
        
        Returns
        -------
        clipped : ndarray, shape (pop_size, dim)
            Clipped vectors
        """
        return np.clip(vectors, self.lb, self.ub)
