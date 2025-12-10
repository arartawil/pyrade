"""
jDE (Self-adaptive Differential Evolution)

Reference:
    Brest, J., Greiner, S., Boskovic, B., Mernik, M., & Zumer, V. (2006).
    Self-adapting control parameters in differential evolution: A comparative study on numerical benchmark problems.
    IEEE Transactions on Evolutionary Computation, 10(6), 646-657.
"""

import numpy as np
from typing import Callable, Optional, Tuple
import logging

from ...core.algorithm import DifferentialEvolution

logger = logging.getLogger(__name__)


class jDE(DifferentialEvolution):
    """
    Self-adaptive Differential Evolution (jDE)
    
    Key features:
    - Self-adaptive scaling factor F and crossover rate CR
    - Parameter values evolve with each individual
    - τ1 and τ2 control adaptation probability
    
    Parameters
    ----------
    pop_size : int
        Population size (default: 100)
    F_init : float
        Initial scaling factor (default: 0.5)
    CR_init : float
        Initial crossover rate (default: 0.9)
    F_l : float
        Lower bound for F (default: 0.1)
    F_u : float
        Upper bound for F (default: 0.9)
    tau1 : float
        Probability of adjusting F (default: 0.1)
    tau2 : float
        Probability of adjusting CR (default: 0.1)
    strategy : str
        Mutation strategy (default: 'rand/1/bin')
    """
    
    def __init__(
        self,
        pop_size: int = 100,
        F_init: float = 0.5,
        CR_init: float = 0.9,
        F_l: float = 0.1,
        F_u: float = 0.9,
        tau1: float = 0.1,
        tau2: float = 0.1,
        strategy: str = 'rand/1/bin',
        **kwargs
    ):
        super().__init__(pop_size=pop_size, F=F_init, CR=CR_init, strategy=strategy, **kwargs)
        self.F_l = F_l
        self.F_u = F_u
        self.tau1 = tau1
        self.tau2 = tau2
        
        # Individual parameter arrays
        self.F_values = None
        self.CR_values = None
        
        logger.info(f"jDE initialized with F_l={F_l}, F_u={F_u}, tau1={tau1}, tau2={tau2}")
    
    def _initialize_parameters(self):
        """Initialize parameter arrays for each individual"""
        self.F_values = np.full(self.pop_size, self.F)
        self.CR_values = np.full(self.pop_size, self.CR)
    
    def _adapt_parameters(self, idx: int) -> Tuple[float, float]:
        """
        Adapt F and CR for individual idx
        
        Parameters
        ----------
        idx : int
            Individual index
            
        Returns
        -------
        Tuple[float, float]
            New F and CR values
        """
        # Adapt F with probability tau1
        if np.random.rand() < self.tau1:
            F_new = self.F_l + np.random.rand() * (self.F_u - self.F_l)
        else:
            F_new = self.F_values[idx]
        
        # Adapt CR with probability tau2
        if np.random.rand() < self.tau2:
            CR_new = np.random.rand()
        else:
            CR_new = self.CR_values[idx]
        
        return F_new, CR_new
    
    def optimize(
        self,
        objective_func: Callable[[np.ndarray], float],
        bounds: np.ndarray,
        max_evals: Optional[int] = None,
        max_iter: Optional[int] = None,
        target: Optional[float] = None,
        show_progress: bool = False
    ) -> Tuple[np.ndarray, float]:
        """
        Run jDE optimization
        
        Parameters
        ----------
        objective_func : Callable
            Objective function to minimize
        bounds : np.ndarray
            Variable bounds (dim, 2)
        max_evals : int, optional
            Maximum function evaluations
        max_iter : int, optional
            Maximum iterations
        target : float, optional
            Target fitness value
        show_progress : bool
            Show progress bar
            
        Returns
        -------
        Tuple[np.ndarray, float]
            Best solution and its fitness
        """
        # Initialize population
        self.population.initialize(bounds, objective_func)
        self._initialize_parameters()
        
        # Setup termination
        max_iter = max_iter or 1000
        max_evals = max_evals or (max_iter * self.pop_size)
        
        # Progress tracking
        if show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=max_evals, desc="jDE")
            except ImportError:
                pbar = None
        else:
            pbar = None
        
        n_evals = self.pop_size  # Already evaluated during initialization
        iteration = 0
        
        try:
            while n_evals < max_evals and iteration < max_iter:
                for i in range(self.pop_size):
                    # Adapt parameters for this individual
                    F_trial, CR_trial = self._adapt_parameters(i)
                    
                    # Generate trial vector
                    trial = self._generate_trial(i, F_trial, CR_trial)
                    
                    # Evaluate trial
                    trial_fitness = objective_func(trial)
                    n_evals += 1
                    
                    # Selection
                    if trial_fitness < self.population.fitness[i]:
                        self.population.population[i] = trial
                        self.population.fitness[i] = trial_fitness
                        self.F_values[i] = F_trial
                        self.CR_values[i] = CR_trial
                        
                        # Update best
                        if trial_fitness < self.population.best_fitness:
                            self.population.best_idx = i
                            self.population.best_fitness = trial_fitness
                    
                    if pbar:
                        pbar.update(1)
                    
                    # Check termination
                    if target is not None and self.population.best_fitness <= target:
                        break
                    if n_evals >= max_evals:
                        break
                
                iteration += 1
                
                if target is not None and self.population.best_fitness <= target:
                    logger.info(f"Target {target} reached at iteration {iteration}")
                    break
        
        finally:
            if pbar:
                pbar.close()
        
        best_solution = self.population.population[self.population.best_idx].copy()
        best_fitness = self.population.best_fitness
        
        logger.info(f"jDE completed: {n_evals} evals, best fitness={best_fitness:.6e}")
        
        return best_solution, best_fitness
