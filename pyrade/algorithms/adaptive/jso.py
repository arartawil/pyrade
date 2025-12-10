"""
jSO (j2020 - CEC 2020 Winner)

Reference:
    Brest, J., Maučec, M. S., & Bošković, B. (2020).
    The 100-digit challenge: Algorithm jDE100.
    In 2020 IEEE Congress on Evolutionary Computation (CEC) (pp. 1-8).
"""

import numpy as np
from typing import Callable, Optional, Tuple, List
import logging

from ...core.algorithm import DifferentialEvolution

logger = logging.getLogger(__name__)


class jSO(DifferentialEvolution):
    """
    jSO (j2020) - Winner of CEC 2020 Competition
    
    Key features:
    - Weighted current-to-pbest-w mutation
    - Enhanced archive management
    - Improved parameter adaptation
    - Terminal-based population size reduction
    
    Parameters
    ----------
    pop_size_init : int
        Initial population size (default: 100)
    pop_size_min : int
        Minimum population size (default: 4)
    H : int
        Historical memory size (default: 5)
    p_min : float
        Minimum p value (default: 0.05)
    p_max : float
        Maximum p value (default: 0.20)
    archive_size_ratio : float
        Archive size ratio (default: 2.6)
    freq_aulr : int
        Frequency for AULR reduction (default: 25)
    """
    
    def __init__(
        self,
        pop_size_init: int = 100,
        pop_size_min: int = 4,
        H: int = 5,
        p_min: float = 0.05,
        p_max: float = 0.20,
        archive_size_ratio: float = 2.6,
        freq_aulr: int = 25,
        **kwargs
    ):
        super().__init__(pop_size=pop_size_init, F=0.5, CR=0.5, **kwargs)
        self.pop_size_init = pop_size_init
        self.pop_size_min = pop_size_min
        self.H = H
        self.p_min = p_min
        self.p_max = p_max
        self.archive_size_ratio = archive_size_ratio
        self.freq_aulr = freq_aulr
        
        # Memory
        self.M_F = np.full(H, 0.5)
        self.M_CR = np.full(H, 0.8)
        self.memory_index = 0
        
        # Archive
        self.archive = []
        
        logger.info(f"jSO initialized: pop_init={pop_size_init}, H={H}, freq_aulr={freq_aulr}")
    
    def _adaptive_p(self, n_evals: int, max_evals: int) -> float:
        """Compute adaptive p value"""
        ratio = n_evals / max_evals
        return self.p_min + (self.p_max - self.p_min) * ratio
    
    def _generate_parameters(self) -> Tuple[float, float]:
        """Generate F and CR from historical memory"""
        r = np.random.randint(self.H)
        
        # Cauchy for F
        F = np.random.standard_cauchy() * 0.1 + self.M_F[r]
        F = np.clip(F, 0, 1)
        
        while F <= 0:
            F = np.random.standard_cauchy() * 0.1 + self.M_F[r]
            F = np.clip(F, 0, 1)
        
        # Normal for CR
        CR = np.random.normal(self.M_CR[r], 0.1)
        CR = np.clip(CR, 0, 1)
        
        return F, CR
    
    def _weighted_current_to_pbest_mutation(
        self,
        idx: int,
        F: float,
        p: float
    ) -> np.ndarray:
        """
        Weighted current-to-pbest-w/1 mutation
        
        Parameters
        ----------
        idx : int
            Target index
        F : float
            Scaling factor
        p : float
            Percentage for pbest
            
        Returns
        -------
        np.ndarray
            Mutant vector
        """
        pop = self.population.population
        fitness = self.population.fitness
        n_pop, dim = pop.shape
        
        # Select pbest from top p%
        p_size = max(2, int(p * n_pop))
        top_indices = np.argsort(fitness)[:p_size]
        pbest_idx = np.random.choice(top_indices)
        
        # Weight for pbest component
        fw = 0.7 * F + 0.3 * np.random.rand() * F
        
        # Select r1
        candidates = [i for i in range(n_pop) if i != idx and i != pbest_idx]
        r1 = np.random.choice(candidates)
        
        # Select r2 from population + archive
        if len(self.archive) > 0:
            combined_pop = np.vstack([pop, np.array(self.archive)])
        else:
            combined_pop = pop
        
        available_indices = [i for i in range(len(combined_pop)) if i != idx and i != pbest_idx and i != r1]
        r2 = np.random.choice(available_indices)
        
        # Weighted mutation: x_i + fw * (x_pbest - x_i) + F * (x_r1 - x_r2)
        mutant = pop[idx] + fw * (pop[pbest_idx] - pop[idx]) + F * (pop[r1] - combined_pop[r2])
        
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
        """Update historical memory"""
        if len(successful_F) == 0:
            return
        
        S_F = np.array(successful_F)
        S_CR = np.array(successful_CR)
        weights = np.array(improvements)
        weights = weights / np.sum(weights)
        
        # Weighted Lehmer mean for F
        mean_F = np.sum(weights * S_F ** 2) / np.sum(weights * S_F)
        
        # Weighted arithmetic mean for CR
        mean_CR = np.sum(weights * S_CR)
        
        self.M_F[self.memory_index] = mean_F
        self.M_CR[self.memory_index] = mean_CR
        
        self.memory_index = (self.memory_index + 1) % self.H
    
    def _aulr_reduction(self, n_evals: int, max_evals: int, iteration: int) -> int:
        """
        AULR (Ageing-based Upon Less-fit Removal) population reduction
        
        Parameters
        ----------
        n_evals : int
            Current evaluations
        max_evals : int
            Maximum evaluations
        iteration : int
            Current iteration
            
        Returns
        -------
        int
            Target population size
        """
        # Linear reduction plan
        ratio = n_evals / max_evals
        plan_size = int(self.pop_size_min + (self.pop_size_init - self.pop_size_min) * (1 - ratio))
        
        # Only reduce every freq_aulr generations
        if iteration % self.freq_aulr == 0 and plan_size < self.pop_size:
            return plan_size
        
        return self.pop_size
    
    def _reduce_population(self, target_size: int):
        """Reduce population by removing worst individuals"""
        if target_size >= self.pop_size:
            return
        
        sorted_indices = np.argsort(self.population.fitness)
        keep_indices = sorted_indices[:target_size]
        
        self.population.population = self.population.population[keep_indices]
        self.population.fitness = self.population.fitness[keep_indices]
        self.pop_size = target_size
        
        self.population.best_idx = 0
        self.population.best_fitness = self.population.fitness[0]
        
        logger.debug(f"Population reduced to {target_size}")
    
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
        Run jSO optimization
        
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
        self.archive = []
        
        max_iter = max_iter or 1000
        max_evals = max_evals or (max_iter * self.pop_size)
        
        if show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=max_evals, desc="jSO")
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
                
                # Adaptive p
                p = self._adaptive_p(n_evals, max_evals)
                
                for i in range(self.pop_size):
                    # Generate parameters
                    F, CR = self._generate_parameters()
                    
                    # Weighted mutation
                    mutant = self._weighted_current_to_pbest_mutation(i, F, p)
                    
                    # Boundary handling
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
                
                # Update memory
                self._update_memory(successful_F, successful_CR, improvements)
                
                # AULR population reduction
                target_pop_size = self._aulr_reduction(n_evals, max_evals, iteration)
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
        
        logger.info(f"jSO completed: {n_evals} evals, final pop_size={self.pop_size}, best fitness={best_fitness:.6e}")
        
        return best_solution, best_fitness
