"""
SHADE (Success-History Based Adaptive Differential Evolution)

Reference:
    Tanabe, R., & Fukunaga, A. (2013).
    Success-history based parameter adaptation for differential evolution.
    In 2013 IEEE Congress on Evolutionary Computation (pp. 71-78).
"""

import numpy as np
from typing import Callable, Optional, Tuple, List
import logging

from ...core.algorithm import DifferentialEvolution

logger = logging.getLogger(__name__)


class SHADE(DifferentialEvolution):
    """
    Success-History Based Adaptive Differential Evolution (SHADE)
    
    Key features:
    - Historical memory of successful F and CR parameters
    - Weighted Lehmer mean for F adaptation
    - Weighted arithmetic mean for CR adaptation
    - current-to-pbest/1 mutation with archive
    
    Parameters
    ----------
    pop_size : int
        Population size (default: 100)
    H : int
        Historical memory size (default: 100)
    p : float
        Percentage of best individuals for pbest (default: 0.1)
    archive_size_ratio : float
        Archive size as ratio of pop_size (default: 1.0)
    """
    
    def __init__(
        self,
        pop_size: int = 100,
        H: int = 100,
        p: float = 0.1,
        archive_size_ratio: float = 1.0,
        **kwargs
    ):
        super().__init__(pop_size=pop_size, F=0.5, CR=0.5, **kwargs)
        self.H = H
        self.p = p
        self.archive_size_ratio = archive_size_ratio
        self.max_archive_size = int(archive_size_ratio * pop_size)
        
        # Historical memory (initialized to 0.5)
        self.M_F = np.full(H, 0.5)
        self.M_CR = np.full(H, 0.5)
        self.memory_index = 0
        
        # Archive
        self.archive = []
        
        logger.info(f"SHADE initialized with H={H}, p={p}, archive_size={self.max_archive_size}")
    
    def _generate_parameters(self) -> Tuple[float, float]:
        """
        Generate F and CR from historical memory
        
        Returns
        -------
        Tuple[float, float]
            F and CR values
        """
        # Select random memory index
        r = np.random.randint(self.H)
        
        # Generate F from Cauchy distribution
        F = np.random.standard_cauchy() * 0.1 + self.M_F[r]
        F = np.clip(F, 0, 1)
        
        # Regenerate if F is too small
        while F <= 0:
            F = np.random.standard_cauchy() * 0.1 + self.M_F[r]
            F = np.clip(F, 0, 1)
        
        # Generate CR from Normal distribution
        CR = np.random.normal(self.M_CR[r], 0.1)
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
        
        # Select r1 (different from idx and pbest_idx)
        candidates = [i for i in range(n_pop) if i != idx and i != pbest_idx]
        r1 = np.random.choice(candidates)
        
        # Select r2 from population + archive
        if len(self.archive) > 0:
            combined_pop = np.vstack([pop, np.array(self.archive)])
        else:
            combined_pop = pop
        
        # r2 must be different from idx, pbest_idx, and r1
        available_indices = [i for i in range(len(combined_pop)) if i != idx and i != pbest_idx and i != r1]
        r2 = np.random.choice(available_indices)
        
        # Mutation
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
    
    def _update_memory(
        self,
        successful_F: List[float],
        successful_CR: List[float],
        improvements: List[float]
    ):
        """
        Update historical memory using weighted means
        
        Parameters
        ----------
        successful_F : List[float]
            Successful F values
        successful_CR : List[float]
            Successful CR values
        improvements : List[float]
            Fitness improvements (weights)
        """
        if len(successful_F) == 0:
            return
        
        # Convert to arrays
        S_F = np.array(successful_F)
        S_CR = np.array(successful_CR)
        weights = np.array(improvements)
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Weighted Lehmer mean for F
        mean_F = np.sum(weights * S_F ** 2) / np.sum(weights * S_F)
        
        # Weighted arithmetic mean for CR
        mean_CR = np.sum(weights * S_CR)
        
        # Update memory
        self.M_F[self.memory_index] = mean_F
        self.M_CR[self.memory_index] = mean_CR
        
        # Move to next memory position
        self.memory_index = (self.memory_index + 1) % self.H
    
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
        Run SHADE optimization
        
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
                pbar = tqdm(total=max_evals, desc="SHADE")
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
                improvements = []
                
                for i in range(self.pop_size):
                    # Generate parameters
                    F, CR = self._generate_parameters()
                    
                    # Generate mutant
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
                        # Calculate improvement
                        improvement = self.population.fitness[i] - trial_fitness
                        
                        # Store replaced solution in archive
                        self._update_archive(self.population.population[i])
                        
                        # Replace
                        self.population.population[i] = trial
                        self.population.fitness[i] = trial_fitness
                        
                        # Record successful parameters
                        successful_F.append(F)
                        successful_CR.append(CR)
                        improvements.append(improvement)
                        
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
                
                # Update historical memory
                self._update_memory(successful_F, successful_CR, improvements)
                
                iteration += 1
                
                if target is not None and self.population.best_fitness <= target:
                    logger.info(f"Target {target} reached at iteration {iteration}")
                    break
        
        finally:
            if pbar:
                pbar.close()
        
        best_solution = self.population.population[self.population.best_idx].copy()
        best_fitness = self.population.best_fitness
        
        logger.info(f"SHADE completed: {n_evals} evals, best fitness={best_fitness:.6e}")
        
        return best_solution, best_fitness
