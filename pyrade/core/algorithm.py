"""
Main Differential Evolution algorithm implementation.

This module provides the core DifferentialEvolution class with
fully vectorized operations for high performance.
"""

import numpy as np
import time
from typing import Callable, Optional, Dict, Any

from pyrade.core.population import Population
from pyrade.operators.mutation import DErand1
from pyrade.operators.crossover import BinomialCrossover
from pyrade.operators.selection import GreedySelection


class DifferentialEvolution:
    """
    Main Differential Evolution optimizer with vectorized operations.
    
    This implementation uses aggressive vectorization to process entire
    populations at once, achieving significant performance improvements
    over monolithic implementations.
    
    Features:
    - Fully vectorized (processes entire population at once)
    - Strategy pattern for operators (easy to extend)
    - Professional API (fit/predict style)
    - Progress tracking and callbacks
    
    Parameters
    ----------
    objective_func : callable
        Function to minimize: f(x) -> float
    bounds : tuple or array
        (lower_bound, upper_bound) or [(lb1, ub1), (lb2, ub2), ...]
    mutation : MutationStrategy, optional
        Mutation strategy (default: DE/rand/1 with F=0.8)
    crossover : CrossoverStrategy, optional
        Crossover strategy (default: Binomial with CR=0.9)
    selection : SelectionStrategy, optional
        Selection strategy (default: Greedy)
    pop_size : int, default=50
        Population size
    max_iter : int, default=1000
        Maximum iterations
    seed : int, optional
        Random seed for reproducibility
    verbose : bool, default=False
        Print progress
    callback : callable, optional
        Called after each iteration: callback(iteration, best_fitness, best_solution)
    
    Attributes
    ----------
    best_solution_ : ndarray
        Best solution found
    best_fitness_ : float
        Best fitness value
    history_ : dict
        Optimization history (fitness, time, etc.)
    
    Methods
    -------
    optimize() : dict
        Run optimization and return results
    
    Examples
    --------
    >>> def sphere(x):
    ...     return sum(x**2)
    >>> 
    >>> optimizer = DifferentialEvolution(
    ...     objective_func=sphere,
    ...     bounds=(-100, 100),
    ...     pop_size=50,
    ...     max_iter=1000
    ... )
    >>> result = optimizer.optimize()
    >>> print(f"Best fitness: {result['best_fitness']}")
    """
    
    def __init__(
        self,
        objective_func: Callable[[np.ndarray], float],
        bounds,
        mutation=None,
        crossover=None,
        selection=None,
        pop_size: int = 50,
        max_iter: int = 1000,
        seed: Optional[int] = None,
        verbose: bool = False,
        callback: Optional[Callable] = None
    ):
        """Initialize Differential Evolution optimizer."""
        # Validate inputs
        if not callable(objective_func):
            raise ValueError("objective_func must be callable")
        if pop_size < 4:
            raise ValueError("pop_size must be at least 4")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        
        self.objective_func = objective_func
        self.bounds = bounds
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.seed = seed
        self.verbose = verbose
        self.callback = callback
        
        # Initialize operators with defaults if not provided
        self.mutation = mutation if mutation is not None else DErand1(F=0.8)
        self.crossover = crossover if crossover is not None else BinomialCrossover(CR=0.9)
        self.selection = selection if selection is not None else GreedySelection()
        
        # Infer dimensionality from bounds
        bounds_array = np.array(bounds)
        if bounds_array.ndim == 1:
            # Need to test objective function to get dimensionality
            # For now, raise error - user should specify dimension-aware bounds
            raise ValueError(
                "Cannot infer dimensionality from scalar bounds. "
                "Please provide bounds as [(lb1, ub1), (lb2, ub2), ...] "
                "or specify dimension explicitly."
            )
        self.dim = bounds_array.shape[0]
        
        # Initialize population
        self.population = Population(pop_size, self.dim, bounds, seed)
        
        # Results storage
        self.best_solution_ = None
        self.best_fitness_ = np.inf
        self.history_ = {
            'fitness': [],
            'time': [],
            'iteration': []
        }
    
    def _initialize_population(self):
        """Initialize random population and evaluate fitness."""
        if self.verbose:
            print("Initializing population...")
        
        self.population.initialize_random()
        self.population.evaluate(self.objective_func)
        
        # Store initial best
        self.best_solution_ = self.population.best_vector.copy()
        self.best_fitness_ = self.population.best_fitness
        
        if self.verbose:
            print(f"Initial best fitness: {self.best_fitness_:.6e}")
    
    def _evolve_generation(self):
        """
        Evolve one generation using vectorized operations.
        
        Steps:
        1. Vectorized mutation (all individuals at once)
        2. Boundary repair (vectorized clip)
        3. Vectorized crossover (all at once)
        4. Boundary repair (vectorized clip)
        5. Evaluate all trials
        6. Vectorized selection (all at once)
        7. Update best solution
        
        Returns
        -------
        improved_count : int
            Number of individuals that improved
        """
        pop_vectors = self.population.vectors
        pop_fitness = self.population.fitness
        best_idx = self.population.best_idx
        target_indices = self.population.get_indices()
        
        # Step 1: Vectorized mutation
        mutants = self.mutation.apply(
            pop_vectors, pop_fitness, best_idx, target_indices
        )
        
        # Step 2: Boundary repair (mutation)
        mutants = self.population.clip_to_bounds(mutants)
        
        # Step 3: Vectorized crossover
        trials = self.crossover.apply(pop_vectors, mutants)
        
        # Step 4: Boundary repair (crossover)
        trials = self.population.clip_to_bounds(trials)
        
        # Step 5: Evaluate all trials
        trial_fitness = self.population.evaluate_vectors(trials, self.objective_func)
        
        # Step 6: Vectorized selection
        new_vectors, new_fitness = self.selection.apply(
            pop_vectors, pop_fitness, trials, trial_fitness
        )
        
        # Count improvements
        improved_count = np.sum(new_fitness < pop_fitness)
        
        # Step 7: Update population
        self.population.update(new_vectors, new_fitness)
        
        # Update global best
        if self.population.best_fitness < self.best_fitness_:
            self.best_solution_ = self.population.best_vector.copy()
            self.best_fitness_ = self.population.best_fitness
        
        return improved_count
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run the optimization.
        
        Returns
        -------
        dict with keys:
            'best_solution': Best solution found
            'best_fitness': Best fitness value
            'n_iterations': Number of iterations run
            'history': Optimization history
            'success': Whether optimization succeeded
            'time': Total optimization time
        """
        if self.verbose:
            print("="*70)
            print("Starting Differential Evolution Optimization")
            print("="*70)
            print(f"Population size: {self.pop_size}")
            print(f"Dimensions: {self.dim}")
            print(f"Max iterations: {self.max_iter}")
            print(f"Mutation: {self.mutation.__class__.__name__}")
            print(f"Crossover: {self.crossover.__class__.__name__}")
            print(f"Selection: {self.selection.__class__.__name__}")
            print("="*70)
        
        start_time = time.time()
        
        # Initialize population
        self._initialize_population()
        
        # Store initial history
        self.history_['fitness'].append(self.best_fitness_)
        self.history_['time'].append(time.time() - start_time)
        self.history_['iteration'].append(0)
        
        # Main optimization loop
        for iteration in range(1, self.max_iter + 1):
            iter_start = time.time()
            
            # Evolve one generation
            improved_count = self._evolve_generation()
            
            # Store history
            iter_time = time.time() - iter_start
            self.history_['fitness'].append(self.best_fitness_)
            self.history_['time'].append(time.time() - start_time)
            self.history_['iteration'].append(iteration)
            
            # Print progress
            if self.verbose and (iteration % 10 == 0 or iteration == 1):
                print(
                    f"Iter {iteration:4d} | "
                    f"Best: {self.best_fitness_:.6e} | "
                    f"Improved: {improved_count:2d}/{self.pop_size} | "
                    f"Time: {iter_time:.3f}s"
                )
            
            # Call callback
            if self.callback is not None:
                self.callback(iteration, self.best_fitness_, self.best_solution_)
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print("="*70)
            print("Optimization Complete")
            print(f"Final best fitness: {self.best_fitness_:.6e}")
            print(f"Total time: {total_time:.3f}s")
            print(f"Average time per iteration: {total_time/self.max_iter:.3f}s")
            print("="*70)
        
        return {
            'best_solution': self.best_solution_,
            'best_fitness': self.best_fitness_,
            'n_iterations': self.max_iter,
            'history': self.history_,
            'success': True,
            'time': total_time
        }
