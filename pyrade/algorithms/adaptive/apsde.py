"""
APSDE (Adaptive Parameter Selection Differential Evolution)

Reference:
    Qin, A. K., & Suganthan, P. N. (2005).
    Self-adaptive differential evolution algorithm for numerical optimization.
    In 2005 IEEE Congress on Evolutionary Computation (Vol. 2, pp. 1785-1791).
"""

import numpy as np
from typing import Callable, Optional, Tuple, List
import logging

from ...core.algorithm import DifferentialEvolution

logger = logging.getLogger(__name__)


class APSDE(DifferentialEvolution):
    """
    Adaptive Parameter Selection Differential Evolution
    
    Key features:
    - Fitness-based parameter adaptation
    - Each individual has its own F and CR
    - Better individuals guide parameter selection
    - Parameters inherit from parents or mutate based on fitness
    
    Parameters
    ----------
    pop_size : int
        Population size (default: 100)
    F_min : float
        Minimum F value (default: 0.4)
    F_max : float
        Maximum F value (default: 0.9)
    CR_min : float
        Minimum CR value (default: 0.1)
    CR_max : float
        Maximum CR value (default: 0.9)
    tau : float
        Adaptation probability (default: 0.1)
    """
    
    def __init__(
        self,
        pop_size: int = 100,
        F_min: float = 0.4,
        F_max: float = 0.9,
        CR_min: float = 0.1,
        CR_max: float = 0.9,
        tau: float = 0.1,
        **kwargs
    ):
        super().__init__(pop_size=pop_size, F=0.5, CR=0.5, **kwargs)
        self.F_min = F_min
        self.F_max = F_max
        self.CR_min = CR_min
        self.CR_max = CR_max
        self.tau = tau
        
        # Individual parameters
        self.F_values = None
        self.CR_values = None
        
        logger.info(f"APSDE initialized: F=[{F_min}, {F_max}], CR=[{CR_min}, {CR_max}], tau={tau}")
    
    def _initialize_parameters(self):
        """Initialize random parameters for each individual"""
        self.F_values = np.random.uniform(self.F_min, self.F_max, self.pop_size)
        self.CR_values = np.random.uniform(self.CR_min, self.CR_max, self.pop_size)
    
    def _compute_fitness_rank(self) -> np.ndarray:
        """
        Compute normalized fitness rank for each individual
        
        Returns
        -------
        np.ndarray
            Normalized rank (0 = best, 1 = worst)
        """
        # Rank by fitness (0 = best)
        ranks = np.argsort(np.argsort(self.population.fitness))
        # Normalize to [0, 1]
        normalized_ranks = ranks / (self.pop_size - 1)
        return normalized_ranks
    
    def _adapt_parameters(
        self,
        idx: int,
        fitness_rank: float
    ) -> Tuple[float, float]:
        """
        Adapt F and CR based on fitness rank
        
        Parameters
        ----------
        idx : int
            Individual index
        fitness_rank : float
            Normalized fitness rank (0 = best, 1 = worst)
            
        Returns
        -------
        Tuple[float, float]
            Adapted F and CR values
        """
        # Better individuals (lower rank) have lower adaptation probability
        adapt_prob = self.tau * (1 + fitness_rank)
        
        if np.random.rand() < adapt_prob:
            # Mutate F: better individuals get higher F
            F_new = self.F_min + np.random.rand() * (self.F_max - self.F_min)
            # Bias towards higher F for better individuals
            if fitness_rank < 0.5:
                F_new = (F_new + self.F_max) / 2
        else:
            F_new = self.F_values[idx]
        
        if np.random.rand() < adapt_prob:
            # Mutate CR: better individuals get higher CR
            CR_new = self.CR_min + np.random.rand() * (self.CR_max - self.CR_min)
            # Bias towards higher CR for better individuals
            if fitness_rank < 0.5:
                CR_new = (CR_new + self.CR_max) / 2
        else:
            CR_new = self.CR_values[idx]
        
        return F_new, CR_new
    
    def _select_mutation_strategy(self, fitness_rank: float) -> str:
        """
        Select mutation strategy based on fitness rank
        
        Parameters
        ----------
        fitness_rank : float
            Normalized fitness rank
            
        Returns
        -------
        str
            Selected mutation strategy
        """
        # Better individuals (lower rank) use exploitation
        # Worse individuals use exploration
        if fitness_rank < 0.3:
            # Best 30%: use best-based strategies
            return np.random.choice(['best/1/bin', 'current-to-best/1/bin'])
        elif fitness_rank < 0.7:
            # Middle 40%: balanced strategies
            return np.random.choice(['rand/1/bin', 'current-to-rand/1'])
        else:
            # Worst 30%: exploratory strategies
            return np.random.choice(['rand/2/bin', 'rand/1/exp'])
    
    def _mutate_adaptive(
        self,
        idx: int,
        strategy: str,
        F: float
    ) -> np.ndarray:
        """
        Generate mutant using adaptive strategy selection
        
        Parameters
        ----------
        idx : int
            Target index
        strategy : str
            Mutation strategy
        F : float
            Scaling factor
            
        Returns
        -------
        np.ndarray
            Mutant vector
        """
        pop = self.population.population
        n_pop, dim = pop.shape
        best_idx = self.population.best_idx
        
        candidates = [i for i in range(n_pop) if i != idx]
        
        if strategy == 'rand/1/bin':
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            
        elif strategy == 'rand/2/bin':
            r1, r2, r3, r4, r5 = np.random.choice(candidates, 5, replace=False)
            mutant = pop[r1] + F * (pop[r2] - pop[r3]) + F * (pop[r4] - pop[r5])
            
        elif strategy == 'best/1/bin':
            r1, r2 = np.random.choice(candidates, 2, replace=False)
            mutant = pop[best_idx] + F * (pop[r1] - pop[r2])
            
        elif strategy == 'current-to-best/1/bin':
            r1, r2 = np.random.choice(candidates, 2, replace=False)
            mutant = pop[idx] + F * (pop[best_idx] - pop[idx]) + F * (pop[r1] - pop[r2])
            
        elif strategy == 'current-to-rand/1':
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            K = np.random.rand()
            mutant = pop[idx] + K * (pop[r1] - pop[idx]) + F * (pop[r2] - pop[r3])
            
        elif strategy == 'rand/1/exp':
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            
        else:
            # Default to rand/1/bin
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
        
        return mutant
    
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
        Run APSDE optimization
        
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
        # Initialize
        self.population.initialize(bounds, objective_func)
        self._initialize_parameters()
        
        max_iter = max_iter or 1000
        max_evals = max_evals or (max_iter * self.pop_size)
        
        if show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=max_evals, desc="APSDE")
            except ImportError:
                pbar = None
        else:
            pbar = None
        
        n_evals = self.pop_size
        iteration = 0
        
        try:
            while n_evals < max_evals and iteration < max_iter:
                # Compute fitness ranks
                fitness_ranks = self._compute_fitness_rank()
                
                for i in range(self.pop_size):
                    # Adapt parameters based on fitness rank
                    F_trial, CR_trial = self._adapt_parameters(i, fitness_ranks[i])
                    
                    # Select strategy based on fitness rank
                    strategy = self._select_mutation_strategy(fitness_ranks[i])
                    
                    # Generate mutant
                    mutant = self._mutate_adaptive(i, strategy, F_trial)
                    
                    # Boundary handling
                    mutant = self.boundary_handler.handle(mutant, bounds)
                    
                    # Crossover
                    if 'exp' in strategy:
                        trial = self._crossover_exponential(self.population.population[i], mutant, CR_trial)
                    else:
                        trial = self._crossover(self.population.population[i], mutant, CR_trial)
                    
                    # Evaluate
                    trial_fitness = objective_func(trial)
                    n_evals += 1
                    
                    # Selection
                    if trial_fitness < self.population.fitness[i]:
                        self.population.population[i] = trial
                        self.population.fitness[i] = trial_fitness
                        
                        # Update parameters on success
                        self.F_values[i] = F_trial
                        self.CR_values[i] = CR_trial
                        
                        # Update best
                        if trial_fitness < self.population.best_fitness:
                            self.population.best_idx = i
                            self.population.best_fitness = trial_fitness
                    
                    if pbar:
                        pbar.update(1)
                    
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
        
        logger.info(f"APSDE completed: {n_evals} evals, best fitness={best_fitness:.6e}")
        logger.info(f"Final mean F={np.mean(self.F_values):.3f}, mean CR={np.mean(self.CR_values):.3f}")
        
        return best_solution, best_fitness
    
    def _crossover_exponential(
        self,
        target: np.ndarray,
        mutant: np.ndarray,
        CR: float
    ) -> np.ndarray:
        """
        Exponential crossover
        
        Parameters
        ----------
        target : np.ndarray
            Target vector
        mutant : np.ndarray
            Mutant vector
        CR : float
            Crossover rate
            
        Returns
        -------
        np.ndarray
            Trial vector
        """
        dim = len(target)
        trial = target.copy()
        
        # Start from random position
        n = np.random.randint(dim)
        L = 0
        
        # Copy consecutive components from mutant
        while True:
            trial[n] = mutant[n]
            n = (n + 1) % dim
            L += 1
            
            if np.random.rand() >= CR or L >= dim:
                break
        
        return trial
