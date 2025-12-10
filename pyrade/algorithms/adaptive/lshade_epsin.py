"""
LSHADE-EpSin (LSHADE with Ensemble of Parameters and Sinusoidal Adaptation)

Reference:
    Awad, N. H., Ali, M. Z., Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2017).
    Problem definitions and evaluation criteria for the CEC 2017 special session 
    and competition on single objective real-parameter numerical optimization.
"""

import numpy as np
from typing import Callable, Optional, Tuple, List
import logging

from ...core.algorithm import DifferentialEvolution

logger = logging.getLogger(__name__)


class LSHADEEpSin(DifferentialEvolution):
    """
    LSHADE with Ensemble of Parameters and Sinusoidal Population Size Reduction
    
    Key features:
    - Ensemble of mutation strategies
    - Sinusoidal population size adaptation
    - Enhanced parameter adaptation with restart mechanism
    - Multiple parameter sets with different F and CR ranges
    
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
        Maximum p value (default: 0.2)
    archive_size_ratio : float
        Archive size ratio (default: 2.6)
    """
    
    def __init__(
        self,
        pop_size_init: int = 100,
        pop_size_min: int = 4,
        H: int = 5,
        p_min: float = 0.05,
        p_max: float = 0.2,
        archive_size_ratio: float = 2.6,
        **kwargs
    ):
        super().__init__(pop_size=pop_size_init, F=0.5, CR=0.5, **kwargs)
        self.pop_size_init = pop_size_init
        self.pop_size_min = pop_size_min
        self.H = H
        self.p_min = p_min
        self.p_max = p_max
        self.archive_size_ratio = archive_size_ratio
        
        # Three parameter sets
        self.M_F = [np.full(H, 0.5) for _ in range(3)]
        self.M_CR = [np.full(H, 0.5) for _ in range(3)]
        self.memory_indices = [0, 0, 0]
        
        # Archive
        self.archive = []
        
        logger.info(f"LSHADE-EpSin initialized: pop_init={pop_size_init}, H={H}, 3 parameter sets")
    
    def _compute_population_size(self, n_evals: int, max_evals: int) -> int:
        """
        Compute population size using sinusoidal reduction
        
        Parameters
        ----------
        n_evals : int
            Current evaluations
        max_evals : int
            Maximum evaluations
            
        Returns
        -------
        int
            Target population size
        """
        ratio = n_evals / max_evals
        # Sinusoidal reduction
        size = int(
            self.pop_size_min + 
            (self.pop_size_init - self.pop_size_min) * 
            (1 - ratio + np.sin(2 * np.pi * ratio) / (2 * np.pi))
        )
        return max(self.pop_size_min, size)
    
    def _adaptive_p(self, n_evals: int, max_evals: int) -> float:
        """
        Compute adaptive p value
        
        Parameters
        ----------
        n_evals : int
            Current evaluations
        max_evals : int
            Maximum evaluations
            
        Returns
        -------
        float
            p value
        """
        ratio = n_evals / max_evals
        return self.p_min + (self.p_max - self.p_min) * ratio
    
    def _select_parameter_set(self) -> int:
        """Select one of three parameter sets randomly"""
        return np.random.randint(3)
    
    def _generate_parameters(self, param_set: int) -> Tuple[float, float]:
        """
        Generate F and CR from selected parameter set
        
        Parameters
        ----------
        param_set : int
            Parameter set index (0, 1, or 2)
            
        Returns
        -------
        Tuple[float, float]
            F and CR values
        """
        r = np.random.randint(self.H)
        
        # Generate F
        F = np.random.standard_cauchy() * 0.1 + self.M_F[param_set][r]
        F = np.clip(F, 0, 1)
        
        while F <= 0:
            F = np.random.standard_cauchy() * 0.1 + self.M_F[param_set][r]
            F = np.clip(F, 0, 1)
        
        # Generate CR
        CR = np.random.normal(self.M_CR[param_set][r], 0.1)
        CR = np.clip(CR, 0, 1)
        
        return F, CR
    
    def _current_to_pbest_mutation(
        self,
        idx: int,
        F: float,
        p: float
    ) -> np.ndarray:
        """
        Current-to-pbest/1 mutation
        
        Parameters
        ----------
        idx : int
            Target index
        F : float
            Scaling factor
        p : float
            Percentage for pbest selection
            
        Returns
        -------
        np.ndarray
            Mutant vector
        """
        pop = self.population.population
        fitness = self.population.fitness
        n_pop, dim = pop.shape
        
        p_size = max(2, int(p * n_pop))
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
    
    def _update_archive(self, replaced_solution: np.ndarray):
        """Add replaced solution to archive"""
        self.archive.append(replaced_solution.copy())
        
        max_archive_size = int(self.archive_size_ratio * self.pop_size)
        if len(self.archive) > max_archive_size:
            remove_idx = np.random.randint(len(self.archive))
            self.archive.pop(remove_idx)
    
    def _update_memory(
        self,
        param_set: int,
        successful_F: List[float],
        successful_CR: List[float],
        improvements: List[float]
    ):
        """
        Update memory for specified parameter set
        
        Parameters
        ----------
        param_set : int
            Parameter set index
        successful_F : List[float]
            Successful F values
        successful_CR : List[float]
            Successful CR values
        improvements : List[float]
            Fitness improvements
        """
        if len(successful_F) == 0:
            return
        
        S_F = np.array(successful_F)
        S_CR = np.array(successful_CR)
        weights = np.array(improvements)
        weights = weights / np.sum(weights)
        
        mean_F = np.sum(weights * S_F ** 2) / np.sum(weights * S_F)
        mean_CR = np.sum(weights * S_CR)
        
        idx = self.memory_indices[param_set]
        self.M_F[param_set][idx] = mean_F
        self.M_CR[param_set][idx] = mean_CR
        
        self.memory_indices[param_set] = (idx + 1) % self.H
    
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
        Run LSHADE-EpSin optimization
        
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
                pbar = tqdm(total=max_evals, desc="LSHADE-EpSin")
            except ImportError:
                pbar = None
        else:
            pbar = None
        
        n_evals = self.pop_size
        iteration = 0
        
        try:
            while n_evals < max_evals and iteration < max_iter:
                # Track successes per parameter set
                successful_params = {0: [], 1: [], 2: []}
                
                # Adaptive p
                p = self._adaptive_p(n_evals, max_evals)
                
                for i in range(self.pop_size):
                    # Select parameter set
                    param_set = self._select_parameter_set()
                    
                    # Generate parameters
                    F, CR = self._generate_parameters(param_set)
                    
                    # Generate mutant
                    mutant = self._current_to_pbest_mutation(i, F, p)
                    
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
                        
                        successful_params[param_set].append((F, CR, improvement))
                        
                        if trial_fitness < self.population.best_fitness:
                            self.population.best_idx = i
                            self.population.best_fitness = trial_fitness
                    
                    if pbar:
                        pbar.update(1)
                    
                    if target is not None and self.population.best_fitness <= target:
                        break
                    if n_evals >= max_evals:
                        break
                
                # Update memories for each parameter set
                for param_set in range(3):
                    if successful_params[param_set]:
                        F_vals = [x[0] for x in successful_params[param_set]]
                        CR_vals = [x[1] for x in successful_params[param_set]]
                        improvements = [x[2] for x in successful_params[param_set]]
                        self._update_memory(param_set, F_vals, CR_vals, improvements)
                
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
        
        logger.info(f"LSHADE-EpSin completed: {n_evals} evals, final pop_size={self.pop_size}, best fitness={best_fitness:.6e}")
        
        return best_solution, best_fitness
