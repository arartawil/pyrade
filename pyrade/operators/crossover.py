"""
Crossover strategies for Differential Evolution.

This module provides crossover strategies with fully vectorized
implementations for high performance.
"""

from abc import ABC, abstractmethod
import numpy as np


class CrossoverStrategy(ABC):
    """
    Abstract base class for crossover strategies.
    
    All crossover strategies should inherit from this class and implement
    the apply() method.
    """
    
    @abstractmethod
    def apply(self, population, mutants):
        """
        Apply crossover between population and mutants.
        
        Parameters
        ----------
        population : ndarray, shape (pop_size, dim)
            Target vectors
        mutants : ndarray, shape (pop_size, dim)
            Mutant vectors
            
        Returns
        -------
        trials : ndarray, shape (pop_size, dim)
            Trial vectors
        """
        pass


class BinomialCrossover(CrossoverStrategy):
    """
    Binomial crossover: u_ij = v_ij if rand() <= CR or j == j_rand, else x_ij
    
    Most common crossover in DE. Each dimension is independently crossed
    with probability CR.
    
    Parameters
    ----------
    CR : float, default=0.9
        Crossover probability (0 <= CR <= 1)
    
    Notes
    -----
    Higher CR values lead to more exploitation (more from mutant),
    lower CR values lead to more exploration (more from parent).
    At least one dimension is always crossed over to ensure the trial
    differs from the target.
    """
    
    def __init__(self, CR=0.9):
        if not 0 <= CR <= 1:
            raise ValueError("CR must be in [0, 1]")
        self.CR = CR
    
    def apply(self, population, mutants):
        """Apply binomial crossover (fully vectorized)."""
        pop_size, dim = population.shape
        
        # Vectorized: generate crossover mask for entire population
        crossover_mask = np.random.rand(pop_size, dim) <= self.CR
        
        # Ensure at least one dimension crosses over per individual
        j_rand = np.random.randint(0, dim, pop_size)
        crossover_mask[np.arange(pop_size), j_rand] = True
        
        # Vectorized crossover
        trials = np.where(crossover_mask, mutants, population)
        return trials


class ExponentialCrossover(CrossoverStrategy):
    """
    Exponential crossover: copies contiguous segment from mutant.
    
    Alternative to binomial crossover. Copies a contiguous segment
    of dimensions from the mutant vector.
    
    Parameters
    ----------
    CR : float, default=0.9
        Crossover probability (0 <= CR <= 1)
    
    Notes
    -----
    Exponential crossover tends to preserve building blocks better
    than binomial crossover. The length of the copied segment follows
    a geometric distribution with parameter CR.
    """
    
    def __init__(self, CR=0.9):
        if not 0 <= CR <= 1:
            raise ValueError("CR must be in [0, 1]")
        self.CR = CR
    
    def apply(self, population, mutants):
        """Apply exponential crossover (fully vectorized)."""
        pop_size, dim = population.shape
        
        # Start with target vectors
        trials = population.copy()
        
        # For each individual, determine crossover segment
        for i in range(pop_size):
            # Random starting position
            n = np.random.randint(0, dim)
            
            # Copy at least one dimension
            trials[i, n] = mutants[i, n]
            
            # Continue copying with probability CR
            L = 1
            while L < dim and np.random.rand() <= self.CR:
                n = (n + 1) % dim  # Wrap around
                trials[i, n] = mutants[i, n]
                L += 1
        
        return trials


class UniformCrossover(CrossoverStrategy):
    """
    Uniform crossover: each dimension independently with probability 0.5.
    
    A simple crossover strategy where each dimension has equal probability
    of coming from either parent or mutant.
    
    Notes
    -----
    This is a special case of binomial crossover with CR=0.5,
    but ensures at least one dimension crosses over.
    """
    
    def __init__(self):
        pass
    
    def apply(self, population, mutants):
        """Apply uniform crossover (fully vectorized)."""
        pop_size, dim = population.shape
        
        # Vectorized: 50% chance for each dimension
        crossover_mask = np.random.rand(pop_size, dim) <= 0.5
        
        # Ensure at least one dimension crosses over per individual
        j_rand = np.random.randint(0, dim, pop_size)
        crossover_mask[np.arange(pop_size), j_rand] = True
        
        # Vectorized crossover
        trials = np.where(crossover_mask, mutants, population)
        return trials


class ArithmeticCrossover(CrossoverStrategy):
    """
    Arithmetic crossover: weighted linear combination of parent and mutant.
    
    Creates trial vectors as a weighted average: trial = alpha * mutant + (1-alpha) * parent
    This creates offspring that lie on a line between parent and mutant vectors.
    
    Parameters
    ----------
    alpha : float, default=0.5
        Weighting factor (0 <= alpha <= 1)
        alpha=0.5 means equal weight to both
        alpha=1.0 means take mutant entirely
        alpha=0.0 means take parent entirely
    adaptive : bool, default=False
        If True, alpha is randomly sampled for each individual
    
    Notes
    -----
    Arithmetic crossover is useful for:
    - Real-valued optimization (continuous domains)
    - Maintaining feasibility when parents are feasible
    - Smoother exploration of search space
    - Better preservation of numerical properties
    
    Examples
    --------
    >>> crossover = ArithmeticCrossover(alpha=0.5)  # Equal blending
    >>> crossover = ArithmeticCrossover(alpha=0.7, adaptive=True)  # Adaptive
    """
    
    def __init__(self, alpha=0.5, adaptive=False):
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha
        self.adaptive = adaptive
    
    def apply(self, population, mutants):
        """Apply arithmetic crossover (fully vectorized)."""
        pop_size, dim = population.shape
        
        if self.adaptive:
            # Random alpha for each individual
            alphas = np.random.uniform(0, 1, (pop_size, 1))
            trials = alphas * mutants + (1 - alphas) * population
        else:
            # Fixed alpha for all
            trials = self.alpha * mutants + (1 - self.alpha) * population
        
        return trials


class ThreePointCrossover(CrossoverStrategy):
    """
    Three-point crossover: exchanges three segments between parent and mutant.
    
    Randomly selects three crossover points and alternates between parent and
    mutant vectors. This creates more diverse offspring than single-point crossover.
    
    Process:
    1. Select three random positions
    2. Segment: [0:p1] from one, [p1:p2] from other, [p2:p3] from first, [p3:] from other
    
    Parameters
    ----------
    None
    
    Notes
    -----
    Three-point crossover is useful for:
    - Maintaining building blocks of intermediate size
    - More diversity than two-point crossover
    - Better mixing of parent and mutant characteristics
    - Discrete optimization problems
    
    The crossover ensures good mixing while preserving some contiguous
    segments from both parents.
    
    Examples
    --------
    >>> crossover = ThreePointCrossover()
    >>> trials = crossover.apply(population, mutants)
    """
    
    def __init__(self):
        pass
    
    def apply(self, population, mutants):
        """Apply three-point crossover."""
        pop_size, dim = population.shape
        
        if dim < 4:
            # For very low dimensions, fall back to binomial
            crossover_mask = np.random.rand(pop_size, dim) <= 0.5
            j_rand = np.random.randint(0, dim, pop_size)
            crossover_mask[np.arange(pop_size), j_rand] = True
            return np.where(crossover_mask, mutants, population)
        
        trials = population.copy()
        
        for i in range(pop_size):
            # Select three random crossover points
            points = sorted(np.random.choice(dim - 1, size=3, replace=False) + 1)
            p1, p2, p3 = points
            
            # Alternate between parent and mutant
            # Segment 1: [0:p1] from mutant
            trials[i, :p1] = mutants[i, :p1]
            # Segment 2: [p1:p2] from parent (keep as is)
            # Segment 3: [p2:p3] from mutant
            trials[i, p2:p3] = mutants[i, p2:p3]
            # Segment 4: [p3:] from parent (keep as is)
        
        return trials
