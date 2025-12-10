"""
L-SHADE (Success-History Based Adaptive DE with Linear Population Size Reduction)

Reference:
    Tanabe, R., & Fukunaga, A. S. (2014).
    Improving the search performance of SHADE using linear population size reduction.
    In 2014 IEEE Congress on Evolutionary Computation (CEC) (pp. 1658-1665).
"""

import numpy as np
from typing import Callable, Optional, Tuple, List
import logging

from ...core.algorithm import DifferentialEvolution

logger = logging.getLogger(__name__)


class LSHADE(DifferentialEvolution):
    """
    Linear Population Size Reduction SHADE (L-SHADE)
    
    Key features:
    - SHADE with linear population size reduction
    - Population reduced gradually during optimization
    - Worst individuals removed when population reduced
    - Maintains diversity while improving convergence
    
    Parameters
    ----------
    pop_size_init : int
        Initial population size (default: 100)
    pop_size_min : int
        Minimum population size (default: 4)
    H : int
        Historical memory size (default: 100)
    p : float
        Percentage of best individuals for pbest (default: 0.1)
    archive_size_ratio : float
        Archive size as ratio of current pop_size (default: 1.0)
    """
    
    def __init__(
        self,
        pop_size_init: int = 100,
        pop_size_min: int = 4,
        H: int = 100,
        p: float = 0.1,
        archive_size_ratio: float = 1.0,
        **kwargs
    ):
        super().__init__(pop_size=pop_size_init, F=0.5, CR=0.5, **kwargs)
        self.pop_size_init = pop_size_init
        self.pop_size_min = pop_size_min
        self.H = H
        self.p = p
        self.archive_size_ratio = archive_size_ratio
        
        # Historical memory
        self.M_F = np.full(H, 0.5)
        self.M_CR = np.full(H, 0.5)
        self.memory_index = 0
        
        # Archive
        self.archive = []
        
        logger.info(f"L-SHADE initialized: pop_init={pop_size_init}, pop_min={pop_size_min}, H={H}")
    
    def _compute_population_size(self, n_evals: int, max_evals: int) -> int:
        """
        Compute population size using linear reduction
        
        Parameters
        ----------
        n_evals : int
            Current number of evaluations
        max_evals : int
            Maximum evaluations
            
        Returns
        -------
        int
            Target population size
        """
        ratio = n_evals / max_evals
        size = int(self.pop_size_min + (self.pop_size_init - self.pop_size_min) * (1 - ratio))
        return max(self.pop_size_min, size)
    
    def _reduce_population(self, target_size: int):
        """
        Reduce population by removing worst individuals
        
        Parameters
        ----------
        target_size : int
            Target population size
        """
        if target_size >= self.pop_size:
            return
        
        # Sort by fitness and keep best individuals
        sorted_indices = np.argsort(self.population.fitness)
        keep_indices = sorted_indices[:target_size]
        
        self.population.population = self.population.population[keep_indices]
        self.population.fitness = self.population.fitness[keep_indices]
        self.pop_size = target_size
        
        # Update best index
        self.population.best_idx = 0
        self.population.best_fitness = self.population.fitness[0]
        
        logger.debug(f"Population reduced to {target_size}")
    
    def _generate_parameters(self) -> Tuple[float, float]:
        """Generate F and CR from historical memory"""
        r = np.random.randint(self.H)
        
        F = np.random.standard_cauchy() * 0.1 + self.M_F[r]
        F = np.clip(F, 0, 1)
        
        while F <= 0:
            F = np.random.standard_cauchy() * 0.1 + self.M_F[r]
            F = np.clip(F, 0, 1)
        
        CR = np.random.normal(self.M_CR[r], 0.1)
        CR = np.clip(CR, 0, 1)
        
        return F, CR
    
    def _current_to_pbest_mutation(self, idx: int, F: float) -> np.ndarray:
        """Current-to-pbest/1 mutation with archive"""
        pop = self.population.population
        fitness = self.population.fitness
        n_pop, dim = pop.shape
        
        p_size = max(2, int(self.p * n_pop))
        top_indices = np.argsort(fitness)[:p_size]
        pbest_idx = np.random.choice(top_indices)
        
        candidates = [i for i in range(n_pop) if i != idx and i != pbest_idx]
        r1 = np.random.choice(candidates)
        
        if len(self.archive) > 0:
            combined_pop = np.vstack([pop, np.array(self.archive)])
        else:
            combined_pop = pop
        
        available_indices = [i for i in range(len(combined_pop)) if i != idx and i != pbest_idx and i != r1]
        r2 = np.random.choice(available_indices)
        
        mutant = pop[idx] + F * (pop[pbest_idx] - pop[idx]) + F * (pop[r1] - combined_pop[r2])
        
        return mutant
    
    def _update_archive(self, replaced_solution: np.ndarray):
        """Add replaced solution to archive"""
        self.archive.append(replaced_solution.copy())
        
        max_archive_size = int(self.archive_size_ratio * self.pop_size)
        if len(self.archive) > max_archive_size:
            remove_idx = np.random.randint(len(self.archive))
            self.archive.pop(remove_idx)
    
    def _update_memory(
        self,
        successful_F: List[float],
        successful_CR: List[float],
        improvements: List[float]
    ):
        """Update historical memory using weighted means"""
        if len(successful_F) == 0:
            return
        
        S_F = np.array(successful_F)
        S_CR = np.array(successful_CR)
        weights = np.array(improvements)
        weights = weights / np.sum(weights)
        
        mean_F = np.sum(weights * S_F ** 2) / np.sum(weights * S_F)
        mean_CR = np.sum(weights * S_CR)
        
        self.M_F[self.memory_index] = mean_F
        self.M_CR[self.memory_index] = mean_CR
        
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
        Run L-SHADE optimization
        
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
                pbar = tqdm(total=max_evals, desc="L-SHADE")
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
                        improvement = self.population.fitness[i] - trial_fitness
                        
                        self._update_archive(self.population.population[i])
                        
                        self.population.population[i] = trial
                        self.population.fitness[i] = trial_fitness
                        
                        successful_F.append(F)
                        successful_CR.append(CR)
                        improvements.append(improvement)
                        
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
                
                # Population size reduction
                target_pop_size = self._compute_population_size(n_evals, max_evals)
                if target_pop_size < self.pop_size:
                    self._reduce_population(target_pop_size)
                
                iteration += 1
                
                if target is not None and self.population.best_fitness <= target:
                    logger.info(f"Target {target} reached at iteration {iteration}")
                    break
        
        finally:
            if pbar:
                pbar.close()
        
        best_solution = self.population.population[self.population.best_idx].copy()
        best_fitness = self.population.best_fitness
        
        logger.info(f"L-SHADE completed: {n_evals} evals, final pop_size={self.pop_size}, best fitness={best_fitness:.6e}")
        
        return best_solution, best_fitness
