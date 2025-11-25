"""
Benchmark functions for testing optimization algorithms.

This module provides standard test functions commonly used to
evaluate and compare optimization algorithms.
"""

import numpy as np


class BenchmarkFunction:
    """
    Base class for benchmark functions.
    
    Attributes
    ----------
    dim : int
        Dimensionality of the function
    bounds : tuple
        Search space bounds (lb, ub)
    optimum : float
        Known global optimum value
    optimum_location : ndarray, optional
        Location of global optimum
    """
    
    def __init__(self, dim):
        self.dim = dim
        self.bounds = None
        self.optimum = None
        self.optimum_location = None
    
    def __call__(self, x):
        """Evaluate function at x."""
        raise NotImplementedError
    
    def get_bounds_array(self):
        """Get bounds as array for each dimension."""
        if isinstance(self.bounds, tuple):
            lb, ub = self.bounds
            return [(lb, ub) for _ in range(self.dim)]
        return self.bounds


class Sphere(BenchmarkFunction):
    """
    Sphere function: f(x) = sum(x^2)
    
    Simple unimodal function. Smooth and convex.
    Global minimum at origin.
    
    Parameters
    ----------
    dim : int, default=30
        Dimensionality
    
    Properties
    ----------
    Domain: [-100, 100]^d
    Global minimum: f(0, ..., 0) = 0
    """
    
    def __init__(self, dim=30):
        super().__init__(dim)
        self.bounds = (-100, 100)
        self.optimum = 0.0
        self.optimum_location = np.zeros(dim)
    
    def __call__(self, x):
        """Evaluate sphere function."""
        return np.sum(x**2)


class Rastrigin(BenchmarkFunction):
    """
    Rastrigin function: f(x) = 10*d + sum(x^2 - 10*cos(2*pi*x))
    
    Highly multimodal function with many local minima.
    Tests ability to escape local optima.
    
    Parameters
    ----------
    dim : int, default=30
        Dimensionality
    
    Properties
    ----------
    Domain: [-5.12, 5.12]^d
    Global minimum: f(0, ..., 0) = 0
    """
    
    def __init__(self, dim=30):
        super().__init__(dim)
        self.bounds = (-5.12, 5.12)
        self.optimum = 0.0
        self.optimum_location = np.zeros(dim)
    
    def __call__(self, x):
        """Evaluate Rastrigin function."""
        return 10*self.dim + np.sum(x**2 - 10*np.cos(2*np.pi*x))


class Rosenbrock(BenchmarkFunction):
    """
    Rosenbrock function: f(x) = sum(100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2)
    
    Valley-shaped function. Convergence to global minimum is difficult.
    
    Parameters
    ----------
    dim : int, default=30
        Dimensionality
    
    Properties
    ----------
    Domain: [-5, 10]^d
    Global minimum: f(1, ..., 1) = 0
    """
    
    def __init__(self, dim=30):
        super().__init__(dim)
        self.bounds = (-5, 10)
        self.optimum = 0.0
        self.optimum_location = np.ones(dim)
    
    def __call__(self, x):
        """Evaluate Rosenbrock function."""
        return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


class Ackley(BenchmarkFunction):
    """
    Ackley function: multimodal with many local minima.
    
    f(x) = -20*exp(-0.2*sqrt(sum(x^2)/d)) - exp(sum(cos(2*pi*x))/d) + 20 + e
    
    Nearly flat outer region with central peak.
    
    Parameters
    ----------
    dim : int, default=30
        Dimensionality
    
    Properties
    ----------
    Domain: [-32, 32]^d
    Global minimum: f(0, ..., 0) = 0
    """
    
    def __init__(self, dim=30):
        super().__init__(dim)
        self.bounds = (-32, 32)
        self.optimum = 0.0
        self.optimum_location = np.zeros(dim)
    
    def __call__(self, x):
        """Evaluate Ackley function."""
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(2*np.pi*x))
        return -20*np.exp(-0.2*np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e


class Griewank(BenchmarkFunction):
    """
    Griewank function: multimodal with many widespread local minima.
    
    f(x) = sum(x^2)/4000 - prod(cos(x[i]/sqrt(i+1))) + 1
    
    Parameters
    ----------
    dim : int, default=30
        Dimensionality
    
    Properties
    ----------
    Domain: [-600, 600]^d
    Global minimum: f(0, ..., 0) = 0
    """
    
    def __init__(self, dim=30):
        super().__init__(dim)
        self.bounds = (-600, 600)
        self.optimum = 0.0
        self.optimum_location = np.zeros(dim)
    
    def __call__(self, x):
        """Evaluate Griewank function."""
        sum_part = np.sum(x**2) / 4000
        i = np.arange(1, len(x) + 1)
        prod_part = np.prod(np.cos(x / np.sqrt(i)))
        return sum_part - prod_part + 1


class Schwefel(BenchmarkFunction):
    """
    Schwefel function: deceptive multimodal function.
    
    f(x) = 418.9829*d - sum(x * sin(sqrt(abs(x))))
    
    Global minimum is far from second-best local minima.
    
    Parameters
    ----------
    dim : int, default=30
        Dimensionality
    
    Properties
    ----------
    Domain: [-500, 500]^d
    Global minimum: f(420.9687, ..., 420.9687) ≈ 0
    """
    
    def __init__(self, dim=30):
        super().__init__(dim)
        self.bounds = (-500, 500)
        self.optimum = 0.0
        self.optimum_location = np.full(dim, 420.9687)
    
    def __call__(self, x):
        """Evaluate Schwefel function."""
        return 418.9829*self.dim - np.sum(x * np.sin(np.sqrt(np.abs(x))))


class Levy(BenchmarkFunction):
    """
    Levy function: multimodal function.
    
    Parameters
    ----------
    dim : int, default=30
        Dimensionality
    
    Properties
    ----------
    Domain: [-10, 10]^d
    Global minimum: f(1, ..., 1) = 0
    """
    
    def __init__(self, dim=30):
        super().__init__(dim)
        self.bounds = (-10, 10)
        self.optimum = 0.0
        self.optimum_location = np.ones(dim)
    
    def __call__(self, x):
        """Evaluate Levy function."""
        w = 1 + (x - 1) / 4
        
        term1 = np.sin(np.pi * w[0])**2
        term2 = np.sum((w[:-1] - 1)**2 * (1 + 10*np.sin(np.pi*w[:-1] + 1)**2))
        term3 = (w[-1] - 1)**2 * (1 + np.sin(2*np.pi*w[-1])**2)
        
        return term1 + term2 + term3


class Michalewicz(BenchmarkFunction):
    """
    Michalewicz function: multimodal with steep valleys.
    
    f(x) = -sum(sin(x[i]) * sin((i+1)*x[i]^2/pi)^(2*m))
    
    Parameters
    ----------
    dim : int, default=30
        Dimensionality
    m : int, default=10
        Steepness parameter
    
    Properties
    ----------
    Domain: [0, pi]^d
    Global minimum: depends on dimension
    """
    
    def __init__(self, dim=30, m=10):
        super().__init__(dim)
        self.bounds = (0, np.pi)
        self.m = m
        self.optimum = None  # Depends on dimension
        self.optimum_location = None
    
    def __call__(self, x):
        """Evaluate Michalewicz function."""
        i = np.arange(1, len(x) + 1)
        return -np.sum(np.sin(x) * np.sin(i * x**2 / np.pi)**(2*self.m))


class Zakharov(BenchmarkFunction):
    """
    Zakharov function: unimodal function.
    
    f(x) = sum(x^2) + (sum(0.5*i*x))^2 + (sum(0.5*i*x))^4
    
    Parameters
    ----------
    dim : int, default=30
        Dimensionality
    
    Properties
    ----------
    Domain: [-5, 10]^d
    Global minimum: f(0, ..., 0) = 0
    """
    
    def __init__(self, dim=30):
        super().__init__(dim)
        self.bounds = (-5, 10)
        self.optimum = 0.0
        self.optimum_location = np.zeros(dim)
    
    def __call__(self, x):
        """Evaluate Zakharov function."""
        i = np.arange(1, len(x) + 1)
        sum1 = np.sum(x**2)
        sum2 = np.sum(0.5 * i * x)
        return sum1 + sum2**2 + sum2**4
