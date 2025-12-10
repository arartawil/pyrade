"""
SaDE (Self-adaptive Differential Evolution)

Reference:
    Qin, A. K., Huang, V. L., & Suganthan, P. N. (2009).
    Differential evolution algorithm with strategy adaptation for global numerical optimization.
    IEEE Transactions on Evolutionary Computation, 13(2), 398-417.
"""

import numpy as np
from typing import Callable, Optional, Tuple, List
import logging

from ...core.algorithm import DifferentialEvolution

logger = logging.getLogger(__name__)


class SaDE(DifferentialEvolution):
    """
    Self-adaptive Differential Evolution with Strategy Pool
    
    Key features:
    - Pool of 4 mutation strategies with adaptive selection
    - Learning period to assess strategy performance
    - CR values generated from normal distribution
    - Strategy probabilities updated based on success rates
    
    Parameters
    ----------
    pop_size : int
        Population size (default: 100)
    F : float
        Scaling factor (default: 0.5)
    learning_period : int
        Number of generations for learning (default: 50)
    CR_m : float
        Mean CR value (default: 0.5)
    """
    
    def __init__(
        self,
        pop_size: int = 100,
        F: float = 0.5,
        learning_period: int = 50,
        CR_m: float = 0.5,
        **kwargs
    ):
        super().__init__(pop_size=pop_size, F=F, CR=0.5, **kwargs)
        self.learning_period = learning_period
        self.CR_m = CR_m
        
        # Strategy pool: rand/1, rand/2, rand-to-best/2, current-to-rand/1
        self.strategies = ['rand/1/bin', 'rand/2/bin', 'rand-to-best/2/bin', 'current-to-rand/1']
        self.n_strategies = len(self.strategies)
        
        # Strategy probabilities (initially equal)
        self.strategy_probs = np.ones(self.n_strategies) / self.n_strategies
        
        # Success tracking
        self.strategy_success = np.zeros(self.n_strategies)
        self.strategy_attempts = np.zeros(self.n_strategies)
        
        # CR memory for each strategy
        self.CR_memory = [[] for _ in range(self.n_strategies)]
        
        logger.info(f"SaDE initialized with {self.n_strategies} strategies, learning_period={learning_period}")
    
    def _select_strategy(self) -> int:
        """Select a strategy based on current probabilities"""
        return np.random.choice(self.n_strategies, p=self.strategy_probs)
    
    def _generate_CR(self, strategy_idx: int) -> float:
        """
        Generate CR value from normal distribution
        
        Parameters
        ----------
        strategy_idx : int
            Index of selected strategy
            
        Returns
        -------
        float
            CR value
        """
        if len(self.CR_memory[strategy_idx]) > 0:
            CR_m = np.mean(self.CR_memory[strategy_idx])
        else:
            CR_m = self.CR_m
        
        CR = np.random.normal(CR_m, 0.1)
        return np.clip(CR, 0, 1)
    
    def _update_strategy_probs(self, generation: int):
        """
        Update strategy probabilities based on success rates
        
        Parameters
        ----------
        generation : int
            Current generation number
        """
        if generation < self.learning_period:
            return
        
        # Calculate success rates
        success_rates = np.zeros(self.n_strategies)
        for i in range(self.n_strategies):
            if self.strategy_attempts[i] > 0:
                success_rates[i] = self.strategy_success[i] / self.strategy_attempts[i]
        
        # Update probabilities
        total_success = np.sum(success_rates)
        if total_success > 0:
            self.strategy_probs = success_rates / total_success
        else:
            self.strategy_probs = np.ones(self.n_strategies) / self.n_strategies
        
        # Ensure minimum probability
        self.strategy_probs = np.maximum(self.strategy_probs, 0.01)
        self.strategy_probs /= np.sum(self.strategy_probs)
        
        # Reset counters
        self.strategy_success = np.zeros(self.n_strategies)
        self.strategy_attempts = np.zeros(self.n_strategies)
    
    def _mutate_with_strategy(
        self,
        idx: int,
        strategy: str,
        F: float
    ) -> np.ndarray:
        """
        Generate mutant vector using specified strategy
        
        Parameters
        ----------
        idx : int
            Target vector index
        strategy : str
            Mutation strategy name
        F : float
            Scaling factor
            
        Returns
        -------
        np.ndarray
            Mutant vector
        """
        pop = self.population.population
        n_pop, dim = pop.shape
        
        # Select random indices (excluding idx)
        candidates = [i for i in range(n_pop) if i != idx]
        
        if strategy == 'rand/1/bin':
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            
        elif strategy == 'rand/2/bin':
            r1, r2, r3, r4, r5 = np.random.choice(candidates, 5, replace=False)
            mutant = pop[r1] + F * (pop[r2] - pop[r3]) + F * (pop[r4] - pop[r5])
            
        elif strategy == 'rand-to-best/2/bin':
            best_idx = self.population.best_idx
            r1, r2, r3, r4 = np.random.choice(candidates, 4, replace=False)
            mutant = pop[r1] + F * (pop[best_idx] - pop[r1]) + F * (pop[r2] - pop[r3]) + F * (pop[r4] - pop[r3])
            
        elif strategy == 'current-to-rand/1':
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            K = np.random.rand()
            mutant = pop[idx] + K * (pop[r1] - pop[idx]) + F * (pop[r2] - pop[r3])
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
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
        Run SaDE optimization
        
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
        
        # Setup termination
        max_iter = max_iter or 1000
        max_evals = max_evals or (max_iter * self.pop_size)
        
        # Progress tracking
        if show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=max_evals, desc="SaDE")
            except ImportError:
                pbar = None
        else:
            pbar = None
        
        n_evals = self.pop_size
        iteration = 0
        
        try:
            while n_evals < max_evals and iteration < max_iter:
                for i in range(self.pop_size):
                    # Select strategy
                    strategy_idx = self._select_strategy()
                    strategy = self.strategies[strategy_idx]
                    
                    # Generate CR
                    CR = self._generate_CR(strategy_idx)
                    
                    # Generate mutant
                    mutant = self._mutate_with_strategy(i, strategy, self.F)
                    
                    # Apply boundary handling
                    mutant = self.boundary_handler.handle(mutant, bounds)
                    
                    # Crossover
                    trial = self._crossover(self.population.population[i], mutant, CR)
                    
                    # Evaluate
                    trial_fitness = objective_func(trial)
                    n_evals += 1
                    
                    # Track strategy attempts
                    self.strategy_attempts[strategy_idx] += 1
                    
                    # Selection
                    if trial_fitness < self.population.fitness[i]:
                        self.population.population[i] = trial
                        self.population.fitness[i] = trial_fitness
                        
                        # Track success
                        self.strategy_success[strategy_idx] += 1
                        self.CR_memory[strategy_idx].append(CR)
                        
                        # Limit memory size
                        if len(self.CR_memory[strategy_idx]) > 25:
                            self.CR_memory[strategy_idx].pop(0)
                        
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
                
                # Update strategy probabilities
                self._update_strategy_probs(iteration)
                
                iteration += 1
                
                if target is not None and self.population.best_fitness <= target:
                    logger.info(f"Target {target} reached at iteration {iteration}")
                    break
        
        finally:
            if pbar:
                pbar.close()
        
        best_solution = self.population.population[self.population.best_idx].copy()
        best_fitness = self.population.best_fitness
        
        logger.info(f"SaDE completed: {n_evals} evals, best fitness={best_fitness:.6e}")
        logger.info(f"Final strategy probabilities: {self.strategy_probs}")
        
        return best_solution, best_fitness
