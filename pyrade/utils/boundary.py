"""
Boundary handling strategies for Differential Evolution.

This module provides various methods for handling vectors that
violate search space boundaries.
"""

import numpy as np
from abc import ABC, abstractmethod


class BoundaryHandler(ABC):
    """
    Abstract base class for boundary handling strategies.
    
    All boundary handlers should inherit from this class and implement
    the repair() method.
    """
    
    @abstractmethod
    def repair(self, vectors, lb, ub):
        """
        Repair vectors that violate boundaries.
        
        Parameters
        ----------
        vectors : ndarray, shape (pop_size, dim)
            Vectors to repair
        lb : ndarray, shape (dim,)
            Lower bounds
        ub : ndarray, shape (dim,)
            Upper bounds
        
        Returns
        -------
        repaired : ndarray, shape (pop_size, dim)
            Repaired vectors within bounds
        """
        pass


class ClipBoundary(BoundaryHandler):
    """
    Clip vectors to boundary (most common).
    
    Values below lower bound are set to lower bound.
    Values above upper bound are set to upper bound.
    
    Notes
    -----
    This is the simplest and most commonly used boundary handling
    method. It can cause clustering at boundaries.
    """
    
    def repair(self, vectors, lb, ub):
        """Clip vectors to bounds (fully vectorized)."""
        return np.clip(vectors, lb, ub)


class ReflectBoundary(BoundaryHandler):
    """
    Reflect vectors at boundary.
    
    Values that exceed bounds are reflected back into the search space.
    For example, if x > ub, the repaired value is ub - (x - ub).
    
    Notes
    -----
    Reflection preserves the magnitude of boundary violations and
    can work better than clipping for certain problems.
    """
    
    def repair(self, vectors, lb, ub):
        """Reflect vectors at bounds (fully vectorized)."""
        repaired = vectors.copy()
        
        # Reflect lower bound violations
        mask_lower = repaired < lb
        repaired[mask_lower] = lb + (lb - repaired[mask_lower])
        
        # Reflect upper bound violations
        mask_upper = repaired > ub
        repaired[mask_upper] = ub - (repaired[mask_upper] - ub)
        
        # If reflected value still violates, clip it
        repaired = np.clip(repaired, lb, ub)
        
        return repaired


class RandomBoundary(BoundaryHandler):
    """
    Replace out-of-bounds values with random values.
    
    Values that violate bounds are replaced with random values
    uniformly sampled from the valid range.
    
    Notes
    -----
    This method maintains diversity better than clipping but
    can be disruptive to the search process.
    """
    
    def repair(self, vectors, lb, ub):
        """Replace out-of-bounds values with random values (vectorized)."""
        repaired = vectors.copy()
        
        # Find violations
        mask_lower = repaired < lb
        mask_upper = repaired > ub
        mask_any = mask_lower | mask_upper
        
        # Replace violations with random values
        if np.any(mask_any):
            random_values = np.random.uniform(
                lb, ub, size=repaired.shape
            )
            repaired[mask_any] = random_values[mask_any]
        
        return repaired


class WrapBoundary(BoundaryHandler):
    """
    Wrap vectors around boundaries (toroidal topology).
    
    Values that exceed bounds wrap around to the other side,
    creating a toroidal search space.
    
    Notes
    -----
    Useful for periodic problems. Can work well for certain
    types of optimization problems with cyclic structure.
    """
    
    def repair(self, vectors, lb, ub):
        """Wrap vectors around bounds (fully vectorized)."""
        range_width = ub - lb
        repaired = vectors.copy()
        
        # Normalize to [0, range_width]
        normalized = repaired - lb
        
        # Apply modulo to wrap
        wrapped = normalized % range_width
        
        # Denormalize back
        repaired = wrapped + lb
        
        return repaired


class MidpointBoundary(BoundaryHandler):
    """
    Set out-of-bounds values to midpoint between bound and parent.
    
    Values that violate bounds are set to the midpoint between
    the violated bound and the corresponding parent vector value.
    
    Parameters
    ----------
    parent_population : ndarray, optional
        Parent population for computing midpoints
    
    Notes
    -----
    This method requires knowledge of parent vectors and can
    help maintain search direction while enforcing bounds.
    """
    
    def __init__(self):
        self.parent_population = None
    
    def set_parent_population(self, parent_population):
        """Set parent population for midpoint calculation."""
        self.parent_population = parent_population
    
    def repair(self, vectors, lb, ub):
        """Set violations to midpoint (vectorized)."""
        if self.parent_population is None:
            # Fallback to clipping if no parent available
            return np.clip(vectors, lb, ub)
        
        repaired = vectors.copy()
        
        # Lower bound violations
        mask_lower = repaired < lb
        if np.any(mask_lower):
            repaired[mask_lower] = (lb + self.parent_population[mask_lower]) / 2
        
        # Upper bound violations
        mask_upper = repaired > ub
        if np.any(mask_upper):
            repaired[mask_upper] = (ub + self.parent_population[mask_upper]) / 2
        
        return repaired
