"""
JADE (Adaptive Differential Evolution with Optional External Archive)

Reference:
    Zhang, J., & Sanderson, A. C. (2009).
    JADE: Adaptive differential evolution with optional external archive.
    IEEE Transactions on Evolutionary Computation, 13(5), 945-958.
"""

import numpy as np
from typing import Callable, Optional, Tuple, List
import logging

from ...core.algorithm import DifferentialEvolution

logger = logging.getLogger(__name__)


class JADE(DifferentialEvolution):
    """
    JADE - Adaptive Differential Evolution with Optional External Archive
    
    Key features:
    - current-to-pbest/1 mutation with archive
    - Adaptive F and CR using Lehmer mean
    - External archive stores recently replaced solutions
    - Parameter adaptation without explicit history memory
    
    Parameters
    ----------
    pop_size : int
        Population size (default: 100)
    c : float
        Learning rate for parameter adaptation (default: 0.1)
    p : float
        Percentage of best individuals for pbest selection (default: 0.05)
    archive_size_ratio : float
        Archive size as ratio of pop_size (default: 1.0)
    mu_F : float
        Initial mean of F (default: 0.5)
    mu_CR : float
        Initial mean of CR (default: 0.5)
    """
    
    def __init__(
        self,
        pop_size: int = 100,
        c: float = 0.1,
        p: float = 0.05,
        archive_size_ratio: float = 1.0,
        mu_F: float = 0.5,
        mu_CR: float = 0.5,
        **kwargs
    ):
        super().__init__(pop_size=pop_size, F=mu_F, CR=mu_CR, **kwargs)
        self.c = c
        self.p = p
        self.archive_size_ratio = archive_size_ratio
        self.max_archive_size = int(archive_size_ratio * pop_size)
        
        # Parameter means
        self.mu_F = mu_F
        self.mu_CR = mu_CR
        
        # Archive for replaced solutions
        self.archive = []
        
        logger.info(f"JADE initialized with c={c}, p={p}, archive_size={self.max_archive_size}")
    
    def _generate_parameters(self) -> Tuple[float, float]:
        """
        Generate F and CR from adapted distributions
        
        Returns
        -------
        Tuple[float, float]
            F and CR values
        """
        # Generate F from Cauchy distribution
        F = np.random.standard_cauchy() * 0.1 + self.mu_F
        F = np.clip(F, 0, 1)
        
        # Regenerate if F is too small
        while F <= 0:
            F = np.random.standard_cauchy() * 0.1 + self.mu_F
            F = np.clip(F, 0, 1)
        
        # Generate CR from Normal distribution
        CR = np.random.normal(self.mu_CR, 0.1)
        CR = np.clip(CR, 0, 1)
        
        return F, CR
    
    def _current_to_pbest_mutation(
        self,
        idx: int,
        F: float
    ) -> np.ndarray:
        """
        Current-to-pbest/1 mutation with archive
        
        Parameters
        ----------
        idx : int
            Target vector index
        F : float
            Scaling factor
            
        Returns
        -------
        np.ndarray
            Mutant vector
        """
        pop = self.population.population
        fitness = self.population.fitness
        n_pop, dim = pop.shape
        
        # Select pbest from top p% individuals
        p_size = max(2, int(self.p * n_pop))
        top_indices = np.argsort(fitness)[:p_size]
        pbest_idx = np.random.choice(top_indices)
        
        # Select random individual r1 (different from idx)
        candidates = [i for i in range(n_pop) if i != idx]
        r1 = np.random.choice(candidates)
        
        # Select r2 from population + archive
        if len(self.archive) > 0:
            combined_pop = np.vstack([pop, np.array(self.archive)])
        else:
            combined_pop = pop
        
        # r2 must be different from idx and r1
        available_indices = [i for i in range(len(combined_pop)) if i != idx and i != r1]
        r2 = np.random.choice(available_indices)
        
        # Mutation: x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)
        mutant = pop[idx] + F * (pop[pbest_idx] - pop[idx]) + F * (pop[r1] - combined_pop[r2])
        
        return mutant
    
    def _update_archive(self, replaced_solution: np.ndarray):
        """
        Add replaced solution to archive
        
        Parameters
        ----------
        replaced_solution : np.ndarray
            Solution that was replaced
        """
        self.archive.append(replaced_solution.copy())
        
        # Remove random solution if archive is full
        if len(self.archive) > self.max_archive_size:
            remove_idx = np.random.randint(len(self.archive))
            self.archive.pop(remove_idx)
    
    def _update_parameters(self, successful_F: List[float], successful_CR: List[float]):
        """
        Update mu_F and mu_CR using Lehmer mean
        
        Parameters
        ----------
        successful_F : List[float]
            F values that led to successful trials
        successful_CR : List[float]
            CR values that led to successful trials
        """
        if len(successful_F) > 0:
            # Lehmer mean for F
            F_array = np.array(successful_F)
            mean_F = np.sum(F_array ** 2) / np.sum(F_array)
            self.mu_F = (1 - self.c) * self.mu_F + self.c * mean_F
        
        if len(successful_CR) > 0:
            # Arithmetic mean for CR
            mean_CR = np.mean(successful_CR)
            self.mu_CR = (1 - self.c) * self.mu_CR + self.c * mean_CR
    
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
        Run JADE optimization
        
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
        self.archive = []
        
        # Setup termination
        max_iter = max_iter or 1000
        max_evals = max_evals or (max_iter * self.pop_size)
        
        # Progress tracking
        if show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=max_evals, desc="JADE")
            except ImportError:
                pbar = None
        else:
            pbar = None
        
        n_evals = self.pop_size
        iteration = 0
        
        try:
            while n_evals < max_evals and iteration < max_iter:
                successful_F = []
                successful_CR = []
                
                for i in range(self.pop_size):
                    # Generate parameters
                    F, CR = self._generate_parameters()
                    
                    # Generate mutant using current-to-pbest/1
                    mutant = self._current_to_pbest_mutation(i, F)
                    
                    # Apply boundary handling
                    mutant = self.boundary_handler.handle(mutant, bounds)
                    
                    # Crossover
                    trial = self._crossover(self.population.population[i], mutant, CR)
                    
                    # Evaluate
                    trial_fitness = objective_func(trial)
                    n_evals += 1
                    
                    # Selection
                    if trial_fitness < self.population.fitness[i]:
                        # Store replaced solution in archive
                        self._update_archive(self.population.population[i])
                        
                        # Replace
                        self.population.population[i] = trial
                        self.population.fitness[i] = trial_fitness
                        
                        # Record successful parameters
                        successful_F.append(F)
                        successful_CR.append(CR)
                        
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
                
                # Update parameter means
                self._update_parameters(successful_F, successful_CR)
                
                iteration += 1
                
                if target is not None and self.population.best_fitness <= target:
                    logger.info(f"Target {target} reached at iteration {iteration}")
                    break
        
        finally:
            if pbar:
                pbar.close()
        
        best_solution = self.population.population[self.population.best_idx].copy()
        best_fitness = self.population.best_fitness
        
        logger.info(f"JADE completed: {n_evals} evals, best fitness={best_fitness:.6e}")
        logger.info(f"Final mu_F={self.mu_F:.3f}, mu_CR={self.mu_CR:.3f}")
        
        return best_solution, best_fitness
