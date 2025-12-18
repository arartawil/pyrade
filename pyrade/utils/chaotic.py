"""
Chaotic Maps for Parameter Control in Differential Evolution.

Chaotic maps generate deterministic sequences with random-like behavior,
useful for dynamic parameter control in DE. They provide better diversity
and exploration compared to purely random or fixed parameters.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Optional


class ChaoticMap(ABC):
    """
    Abstract base class for chaotic maps.
    
    All chaotic maps should inherit from this class and implement
    the next() method to generate the next value in the sequence.
    """
    
    @abstractmethod
    def next(self) -> float:
        """
        Generate next value in the chaotic sequence.
        
        Returns
        -------
        value : float
            Next chaotic value in range [0, 1]
        """
        pass
    
    @abstractmethod
    def reset(self, initial_value: Optional[float] = None):
        """
        Reset the chaotic map to initial state.
        
        Parameters
        ----------
        initial_value : float, optional
            Initial value for the map. If None, uses default.
        """
        pass
    
    def generate_sequence(self, length: int) -> np.ndarray:
        """
        Generate a sequence of chaotic values.
        
        Parameters
        ----------
        length : int
            Length of sequence to generate
            
        Returns
        -------
        sequence : ndarray, shape (length,)
            Chaotic sequence
        """
        return np.array([self.next() for _ in range(length)])


class LogisticMap(ChaoticMap):
    """
    Logistic map: x_{n+1} = r * x_n * (1 - x_n)
    
    The logistic map is one of the most famous chaotic maps, exhibiting
    complex dynamics for certain parameter values. At r=4.0, it displays
    fully chaotic behavior.
    
    Parameters
    ----------
    r : float, default=4.0
        Control parameter (typically 3.57 < r <= 4.0 for chaos)
    initial_value : float, default=0.7
        Initial value (0 < x_0 < 1)
    
    Notes
    -----
    - r < 3.57: Periodic behavior
    - r > 3.57: Chaotic behavior  
    - r = 4.0: Fully chaotic
    
    References
    ----------
    May, R. M. (1976). Simple mathematical models with very complicated
    dynamics. Nature, 261(5560), 459-467.
    
    Examples
    --------
    >>> logistic = LogisticMap(r=4.0)
    >>> sequence = logistic.generate_sequence(100)
    >>> F_values = 0.4 + 0.6 * sequence  # Map to [0.4, 1.0]
    """
    
    def __init__(self, r: float = 4.0, initial_value: float = 0.7):
        if not 0 < initial_value < 1:
            raise ValueError("initial_value must be in (0, 1)")
        if not 0 < r <= 4:
            raise ValueError("r must be in (0, 4]")
        
        self.r = r
        self.initial_value = initial_value
        self.current_value = initial_value
    
    def next(self) -> float:
        """Generate next logistic map value."""
        self.current_value = self.r * self.current_value * (1 - self.current_value)
        
        # Handle edge cases to prevent convergence to 0 or 1
        if self.current_value <= 0.001 or self.current_value >= 0.999:
            self.current_value = np.random.uniform(0.1, 0.9)
        
        return self.current_value
    
    def reset(self, initial_value: Optional[float] = None):
        """Reset to initial state."""
        if initial_value is not None:
            if not 0 < initial_value < 1:
                raise ValueError("initial_value must be in (0, 1)")
            self.initial_value = initial_value
        self.current_value = self.initial_value


class TentMap(ChaoticMap):
    """
    Tent map: x_{n+1} = x_n / 0.5 if x_n < 0.5 else (1 - x_n) / 0.5
    
    The tent map is a piecewise linear chaotic map with uniform invariant
    density. It's simpler than the logistic map but still exhibits chaos.
    
    Parameters
    ----------
    initial_value : float, default=0.7
        Initial value (0 < x_0 < 1)
    
    Notes
    -----
    The tent map always produces chaotic behavior for initial values in (0,1).
    It's computationally faster than the logistic map.
    
    Examples
    --------
    >>> tent = TentMap()
    >>> CR_values = tent.generate_sequence(100)
    """
    
    def __init__(self, initial_value: float = 0.7):
        if not 0 < initial_value < 1:
            raise ValueError("initial_value must be in (0, 1)")
        
        self.initial_value = initial_value
        self.current_value = initial_value
    
    def next(self) -> float:
        """Generate next tent map value."""
        if self.current_value < 0.5:
            self.current_value = self.current_value / 0.5
        else:
            self.current_value = (1 - self.current_value) / 0.5
        
        # Prevent edge cases
        if self.current_value <= 0.001 or self.current_value >= 0.999:
            self.current_value = np.random.uniform(0.1, 0.9)
        
        return self.current_value
    
    def reset(self, initial_value: Optional[float] = None):
        """Reset to initial state."""
        if initial_value is not None:
            if not 0 < initial_value < 1:
                raise ValueError("initial_value must be in (0, 1)")
            self.initial_value = initial_value
        self.current_value = self.initial_value


class SineMap(ChaoticMap):
    """
    Sine map: x_{n+1} = a * sin(π * x_n)
    
    The sine map produces chaotic behavior for a = 4 and exhibits
    ergodic properties useful for parameter control.
    
    Parameters
    ----------
    a : float, default=4.0
        Control parameter (typically a=4 for chaos)
    initial_value : float, default=0.7
        Initial value (0 < x_0 < 1)
    
    Notes
    -----
    The sine map is symmetric and bounded, making it suitable for
    parameter adaptation in optimization algorithms.
    
    Examples
    --------
    >>> sine = SineMap(a=4.0)
    >>> params = 0.5 + 0.5 * sine.generate_sequence(50)
    """
    
    def __init__(self, a: float = 4.0, initial_value: float = 0.7):
        if not 0 < initial_value < 1:
            raise ValueError("initial_value must be in (0, 1)")
        
        self.a = a
        self.initial_value = initial_value
        self.current_value = initial_value
    
    def next(self) -> float:
        """Generate next sine map value."""
        self.current_value = self.a * np.sin(np.pi * self.current_value) / 4.0
        
        # Map to [0, 1] and handle edge cases
        self.current_value = np.clip(self.current_value, 0, 1)
        
        if self.current_value <= 0.001 or self.current_value >= 0.999:
            self.current_value = np.random.uniform(0.1, 0.9)
        
        return self.current_value
    
    def reset(self, initial_value: Optional[float] = None):
        """Reset to initial state."""
        if initial_value is not None:
            if not 0 < initial_value < 1:
                raise ValueError("initial_value must be in (0, 1)")
            self.initial_value = initial_value
        self.current_value = self.initial_value


class ChebyshevMap(ChaoticMap):
    """
    Chebyshev map: x_{n+1} = cos(k * arccos(x_n))
    
    The Chebyshev map of order k is derived from Chebyshev polynomials
    and exhibits chaotic behavior for certain values of k.
    
    Parameters
    ----------
    k : int, default=4
        Order of the Chebyshev polynomial (k >= 2 for chaos)
    initial_value : float, default=0.7
        Initial value (-1 < x_0 < 1)
    
    Notes
    -----
    Higher values of k lead to more complex chaotic behavior. The map
    is defined on [-1, 1] but we rescale to [0, 1] for consistency.
    
    References
    ----------
    Peitgen, H. O., Jürgens, H., & Saupe, D. (2006). Chaos and fractals:
    new frontiers of science. Springer Science & Business Media.
    
    Examples
    --------
    >>> chebyshev = ChebyshevMap(k=5)
    >>> sequence = chebyshev.generate_sequence(100)
    """
    
    def __init__(self, k: int = 4, initial_value: float = 0.7):
        if k < 2:
            raise ValueError("k must be >= 2 for chaotic behavior")
        
        self.k = k
        # Convert from [0,1] to [-1,1] for Chebyshev
        self.initial_value = 2 * initial_value - 1
        self.current_value = self.initial_value
        
        if not -1 < self.current_value < 1:
            raise ValueError("initial_value must lead to value in (-1, 1)")
    
    def next(self) -> float:
        """Generate next Chebyshev map value."""
        # Chebyshev map: T_k(x) = cos(k * arccos(x))
        self.current_value = np.cos(self.k * np.arccos(np.clip(self.current_value, -0.999, 0.999)))
        
        # Handle edge cases
        if abs(self.current_value) >= 0.999:
            self.current_value = np.random.uniform(-0.8, 0.8)
        
        # Convert from [-1,1] to [0,1]
        return (self.current_value + 1) / 2
    
    def reset(self, initial_value: Optional[float] = None):
        """Reset to initial state."""
        if initial_value is not None:
            self.initial_value = 2 * initial_value - 1
            if not -1 < self.initial_value < 1:
                raise ValueError("initial_value must lead to value in (-1, 1)")
        self.current_value = self.initial_value


class ChaoticParameterController:
    """
    Controller for chaotic parameter adaptation in DE.
    
    Uses chaotic maps to dynamically control F and CR parameters during
    optimization, providing better exploration-exploitation balance.
    
    Parameters
    ----------
    chaotic_map : ChaoticMap
        Chaotic map to use for parameter generation
    F_range : tuple, default=(0.4, 1.0)
        Range for mutation factor F: (F_min, F_max)
    CR_range : tuple, default=(0.1, 1.0)
        Range for crossover rate CR: (CR_min, CR_max)
    update_frequency : int, default=1
        Generations between parameter updates (1 = every generation)
    
    Examples
    --------
    >>> # Using Logistic map for F control
    >>> logistic = LogisticMap(r=4.0)
    >>> controller = ChaoticParameterController(
    ...     chaotic_map=logistic,
    ...     F_range=(0.4, 0.9),
    ...     CR_range=(0.1, 0.9)
    ... )
    >>> 
    >>> for generation in range(max_iterations):
    ...     F, CR = controller.get_parameters()
    ...     # Use F and CR in DE operations
    ...     controller.update()
    """
    
    def __init__(
        self,
        chaotic_map: ChaoticMap,
        F_range: tuple = (0.4, 1.0),
        CR_range: tuple = (0.1, 1.0),
        update_frequency: int = 1
    ):
        self.chaotic_map = chaotic_map
        self.F_min, self.F_max = F_range
        self.CR_min, self.CR_max = CR_range
        self.update_frequency = update_frequency
        
        self.generation = 0
        self.current_F = None
        self.current_CR = None
        
        # Initialize parameters
        self._update_parameters()
    
    def _update_parameters(self):
        """Update F and CR using chaotic map."""
        # Get chaotic value
        chaotic_value = self.chaotic_map.next()
        
        # Map to F range
        self.current_F = self.F_min + chaotic_value * (self.F_max - self.F_min)
        
        # Get another chaotic value for CR (advance the map)
        chaotic_value = self.chaotic_map.next()
        self.current_CR = self.CR_min + chaotic_value * (self.CR_max - self.CR_min)
    
    def get_parameters(self) -> tuple:
        """
        Get current F and CR parameters.
        
        Returns
        -------
        F : float
            Current mutation factor
        CR : float
            Current crossover rate
        """
        return self.current_F, self.current_CR
    
    def update(self):
        """Update parameters based on generation count."""
        self.generation += 1
        
        if self.generation % self.update_frequency == 0:
            self._update_parameters()
    
    def reset(self):
        """Reset controller to initial state."""
        self.generation = 0
        self.chaotic_map.reset()
        self._update_parameters()


def create_chaotic_controller(
    map_type: str = 'logistic',
    F_range: tuple = (0.4, 1.0),
    CR_range: tuple = (0.1, 1.0),
    **kwargs
) -> ChaoticParameterController:
    """
    Convenience function to create a chaotic parameter controller.
    
    Parameters
    ----------
    map_type : str, default='logistic'
        Type of chaotic map: 'logistic', 'tent', 'sine', 'chebyshev'
    F_range : tuple
        Range for F parameter
    CR_range : tuple
        Range for CR parameter
    **kwargs : dict
        Additional arguments for the chaotic map
        
    Returns
    -------
    controller : ChaoticParameterController
        Configured chaotic parameter controller
        
    Examples
    --------
    >>> controller = create_chaotic_controller('logistic', F_range=(0.5, 0.9))
    >>> controller = create_chaotic_controller('tent')
    >>> controller = create_chaotic_controller('chebyshev', k=5)
    """
    map_classes = {
        'logistic': LogisticMap,
        'tent': TentMap,
        'sine': SineMap,
        'chebyshev': ChebyshevMap
    }
    
    if map_type not in map_classes:
        raise ValueError(f"map_type must be one of {list(map_classes.keys())}")
    
    chaotic_map = map_classes[map_type](**kwargs)
    
    return ChaoticParameterController(
        chaotic_map=chaotic_map,
        F_range=F_range,
        CR_range=CR_range
    )
