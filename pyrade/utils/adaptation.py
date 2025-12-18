"""
Adaptive mechanisms for Differential Evolution.

This module provides adaptive population sizing and parameter ensemble
strategies for dynamic optimization behavior.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Optional, Union


class AdaptivePopulationSize:
    """
    Adaptive Population Size mechanism for DE.
    
    Dynamically adjusts population size during optimization based on
    convergence progress and diversity metrics. This helps balance
    exploration and exploitation phases.
    
    Strategies:
    -----------
    - 'linear-reduction': Linearly reduce population size over iterations
    - 'success-based': Reduce when success rate is high, increase when low
    - 'diversity-based': Adjust based on population diversity
    - 'lshade-like': Similar to L-SHADE algorithm's population reduction
    
    Parameters
    ----------
    initial_size : int
        Starting population size
    min_size : int
        Minimum population size allowed
    strategy : str, default='linear-reduction'
        Strategy for adapting population size
    reduction_rate : float, default=0.5
        Rate at which to reduce population (for linear strategy)
    diversity_threshold : float, default=0.1
        Diversity threshold for diversity-based strategy
    
    Examples
    --------
    >>> aps = AdaptivePopulationSize(
    ...     initial_size=100,
    ...     min_size=20,
    ...     strategy='linear-reduction'
    ... )
    >>> new_size = aps.update(generation=50, max_generations=1000)
    """
    
    def __init__(
        self,
        initial_size: int,
        min_size: int = 4,
        strategy: str = 'linear-reduction',
        reduction_rate: float = 0.5,
        diversity_threshold: float = 0.1,
    ):
        if initial_size < min_size:
            raise ValueError("initial_size must be >= min_size")
        if min_size < 4:
            raise ValueError("min_size must be >= 4 for DE to work")
        
        self.initial_size = initial_size
        self.min_size = min_size
        self.strategy = strategy
        self.reduction_rate = reduction_rate
        self.diversity_threshold = diversity_threshold
        
        self.current_size = initial_size
        self.success_history = []
        self.diversity_history = []
        
    def update(
        self,
        generation: int,
        max_generations: int,
        population: Optional[np.ndarray] = None,
        fitness: Optional[np.ndarray] = None,
        success_rate: Optional[float] = None,
    ) -> int:
        """
        Update and return the new population size.
        
        Parameters
        ----------
        generation : int
            Current generation number
        max_generations : int
            Maximum number of generations
        population : ndarray, optional
            Current population (needed for diversity-based)
        fitness : ndarray, optional
            Current fitness values
        success_rate : float, optional
            Success rate in last generation (for success-based)
            
        Returns
        -------
        new_size : int
            Updated population size
        """
        if self.strategy == 'linear-reduction':
            new_size = self._linear_reduction(generation, max_generations)
        elif self.strategy == 'lshade-like':
            new_size = self._lshade_reduction(generation, max_generations)
        elif self.strategy == 'success-based':
            new_size = self._success_based(success_rate)
        elif self.strategy == 'diversity-based':
            if population is None:
                raise ValueError("population required for diversity-based strategy")
            new_size = self._diversity_based(population, fitness)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        self.current_size = max(self.min_size, new_size)
        return self.current_size
    
    def _linear_reduction(self, generation: int, max_generations: int) -> int:
        """Linear reduction from initial to minimum size."""
        progress = generation / max_generations
        size_range = self.initial_size - self.min_size
        new_size = int(self.initial_size - size_range * progress * self.reduction_rate)
        return new_size
    
    def _lshade_reduction(self, generation: int, max_generations: int) -> int:
        """
        L-SHADE style reduction: exponential-like decrease.
        
        Formula: N_min + (N_init - N_min) * ((max_gen - gen) / max_gen)^2
        """
        remaining_ratio = (max_generations - generation) / max_generations
        size_range = self.initial_size - self.min_size
        new_size = int(self.min_size + size_range * (remaining_ratio ** 2))
        return new_size
    
    def _success_based(self, success_rate: Optional[float]) -> int:
        """Adapt based on success rate: high success -> reduce, low -> increase."""
        if success_rate is None:
            return self.current_size
        
        self.success_history.append(success_rate)
        
        # Keep last 10 generations
        if len(self.success_history) > 10:
            self.success_history.pop(0)
        
        avg_success = np.mean(self.success_history)
        
        # High success rate (>0.3) -> reduce population
        # Low success rate (<0.1) -> increase population
        if avg_success > 0.3:
            new_size = int(self.current_size * 0.95)
        elif avg_success < 0.1:
            new_size = int(self.current_size * 1.05)
        else:
            new_size = self.current_size
        
        # Constrain to valid range
        new_size = min(new_size, self.initial_size)
        return new_size
    
    def _diversity_based(
        self,
        population: np.ndarray,
        fitness: Optional[np.ndarray] = None
    ) -> int:
        """Adapt based on population diversity."""
        # Calculate diversity as average pairwise distance
        pop_size, dim = population.shape
        
        # Sample-based diversity for efficiency with large populations
        sample_size = min(pop_size, 50)
        indices = np.random.choice(pop_size, sample_size, replace=False)
        sample = population[indices]
        
        # Average pairwise distance (normalized)
        distances = []
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                dist = np.linalg.norm(sample[i] - sample[j]) / np.sqrt(dim)
                distances.append(dist)
        
        diversity = np.mean(distances)
        self.diversity_history.append(diversity)
        
        # Keep last 10 generations
        if len(self.diversity_history) > 10:
            self.diversity_history.pop(0)
        
        avg_diversity = np.mean(self.diversity_history)
        
        # Low diversity -> reduce population (converging)
        # High diversity -> maintain or increase (exploring)
        if avg_diversity < self.diversity_threshold:
            new_size = int(self.current_size * 0.95)
        elif avg_diversity > self.diversity_threshold * 3:
            new_size = int(self.current_size * 1.02)
        else:
            new_size = self.current_size
        
        # Constrain to valid range
        new_size = min(new_size, self.initial_size)
        return new_size
    
    def should_resize(self, current_pop_size: int) -> Tuple[bool, int]:
        """
        Check if population should be resized.
        
        Returns
        -------
        should_resize : bool
            Whether resizing is needed
        target_size : int
            Target population size
        """
        if self.current_size != current_pop_size:
            return True, self.current_size
        return False, current_pop_size
    
    def resize_population(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        target_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resize population to target size.
        
        If reducing: keep best individuals
        If increasing: duplicate with random perturbation
        
        Parameters
        ----------
        population : ndarray, shape (pop_size, dim)
            Current population
        fitness : ndarray, shape (pop_size,)
            Current fitness values
        target_size : int
            Target population size
            
        Returns
        -------
        new_population : ndarray, shape (target_size, dim)
            Resized population
        new_fitness : ndarray, shape (target_size,)
            Resized fitness array
        """
        current_size = len(population)
        
        if target_size == current_size:
            return population, fitness
        
        if target_size < current_size:
            # Reduce: keep best individuals
            best_indices = np.argsort(fitness)[:target_size]
            return population[best_indices], fitness[best_indices]
        else:
            # Increase: duplicate best with perturbation
            new_population = np.zeros((target_size, population.shape[1]))
            new_fitness = np.full(target_size, np.inf)
            
            # Keep current population
            new_population[:current_size] = population
            new_fitness[:current_size] = fitness
            
            # Add new individuals by perturbing best ones
            best_indices = np.argsort(fitness)
            num_to_add = target_size - current_size
            
            for i in range(num_to_add):
                # Select from best individuals (cycling if needed)
                source_idx = best_indices[i % len(best_indices)]
                
                # Add random perturbation (5% of population std)
                std = np.std(population, axis=0)
                perturbation = np.random.normal(0, 0.05 * std, population.shape[1])
                new_population[current_size + i] = population[source_idx] + perturbation
            
            return new_population, new_fitness


class ParameterEnsemble:
    """
    Parameter Ensemble for DE - mix multiple (F, CR) parameter settings.
    
    Maintains a pool of parameter combinations and adaptively selects
    successful ones. This helps the algorithm adapt to different phases
    of optimization and problem landscapes.
    
    Parameters
    ----------
    F_values : list of float, optional
        Mutation factor values to ensemble (default: [0.5, 0.6, 0.7, 0.8, 0.9])
    CR_values : list of float, optional
        Crossover rate values to ensemble (default: [0.1, 0.3, 0.5, 0.7, 0.9])
    strategy : str, default='uniform'
        Selection strategy: 'uniform', 'adaptive', 'random'
    learning_period : int, default=50
        Generations to accumulate success before adapting weights
    
    Examples
    --------
    >>> ensemble = ParameterEnsemble(
    ...     F_values=[0.5, 0.8, 0.9],
    ...     CR_values=[0.1, 0.5, 0.9],
    ...     strategy='adaptive'
    ... )
    >>> F, CR = ensemble.sample(pop_size=50)
    >>> ensemble.update_success(successful_indices, F_indices, CR_indices)
    """
    
    def __init__(
        self,
        F_values: Optional[List[float]] = None,
        CR_values: Optional[List[float]] = None,
        strategy: str = 'uniform',
        learning_period: int = 50,
    ):
        self.F_values = F_values if F_values is not None else [0.5, 0.6, 0.7, 0.8, 0.9]
        self.CR_values = CR_values if CR_values is not None else [0.1, 0.3, 0.5, 0.7, 0.9]
        self.strategy = strategy
        self.learning_period = learning_period
        
        # Initialize uniform weights
        self.F_weights = np.ones(len(self.F_values)) / len(self.F_values)
        self.CR_weights = np.ones(len(self.CR_values)) / len(self.CR_values)
        
        # Success tracking
        self.F_success_counts = np.zeros(len(self.F_values))
        self.F_trial_counts = np.zeros(len(self.F_values))
        self.CR_success_counts = np.zeros(len(self.CR_values))
        self.CR_trial_counts = np.zeros(len(self.CR_values))
        
        self.generation = 0
        
    def sample(self, pop_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample F and CR values for the population.
        
        Parameters
        ----------
        pop_size : int
            Population size
            
        Returns
        -------
        F_array : ndarray, shape (pop_size,)
            Sampled F values
        CR_array : ndarray, shape (pop_size,)
            Sampled CR values
        F_indices : ndarray, shape (pop_size,)
            Indices of F values in ensemble
        CR_indices : ndarray, shape (pop_size,)
            Indices of CR values in ensemble
        """
        if self.strategy == 'uniform':
            # Equal probability for all parameters
            F_indices = np.random.choice(
                len(self.F_values), size=pop_size, replace=True
            )
            CR_indices = np.random.choice(
                len(self.CR_values), size=pop_size, replace=True
            )
        elif self.strategy == 'adaptive':
            # Weighted sampling based on success rates
            F_indices = np.random.choice(
                len(self.F_values), size=pop_size, replace=True, p=self.F_weights
            )
            CR_indices = np.random.choice(
                len(self.CR_values), size=pop_size, replace=True, p=self.CR_weights
            )
        elif self.strategy == 'random':
            # Completely random within reasonable bounds
            F_array = np.random.uniform(0.4, 1.0, pop_size)
            CR_array = np.random.uniform(0.0, 1.0, pop_size)
            return F_array, CR_array, None, None
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        F_array = np.array([self.F_values[i] for i in F_indices])
        CR_array = np.array([self.CR_values[i] for i in CR_indices])
        
        return F_array, CR_array, F_indices, CR_indices
    
    def update_success(
        self,
        successful_indices: np.ndarray,
        F_indices: Optional[np.ndarray],
        CR_indices: Optional[np.ndarray]
    ):
        """
        Update success statistics for parameter combinations.
        
        Parameters
        ----------
        successful_indices : ndarray
            Indices of individuals that improved
        F_indices : ndarray, optional
            F parameter indices used for each individual
        CR_indices : ndarray, optional
            CR parameter indices used for each individual
        """
        if F_indices is None or CR_indices is None:
            return  # Can't update for random strategy
        
        # Update trial counts for all
        for f_idx in F_indices:
            self.F_trial_counts[f_idx] += 1
        for cr_idx in CR_indices:
            self.CR_trial_counts[cr_idx] += 1
        
        # Update success counts for successful ones
        for idx in successful_indices:
            self.F_success_counts[F_indices[idx]] += 1
            self.CR_success_counts[CR_indices[idx]] += 1
        
        self.generation += 1
        
        # Adapt weights every learning_period generations
        if self.generation % self.learning_period == 0:
            self._adapt_weights()
    
    def _adapt_weights(self):
        """Adapt weights based on success rates."""
        # Calculate success rates
        F_success_rates = np.zeros(len(self.F_values))
        for i in range(len(self.F_values)):
            if self.F_trial_counts[i] > 0:
                F_success_rates[i] = self.F_success_counts[i] / self.F_trial_counts[i]
            else:
                F_success_rates[i] = 1.0 / len(self.F_values)  # Default
        
        CR_success_rates = np.zeros(len(self.CR_values))
        for i in range(len(self.CR_values)):
            if self.CR_trial_counts[i] > 0:
                CR_success_rates[i] = self.CR_success_counts[i] / self.CR_trial_counts[i]
            else:
                CR_success_rates[i] = 1.0 / len(self.CR_values)  # Default
        
        # Smooth with previous weights (momentum)
        momentum = 0.7
        new_F_weights = F_success_rates + 0.1  # Add small constant to avoid zeros
        new_F_weights = new_F_weights / np.sum(new_F_weights)
        self.F_weights = momentum * self.F_weights + (1 - momentum) * new_F_weights
        self.F_weights = self.F_weights / np.sum(self.F_weights)
        
        new_CR_weights = CR_success_rates + 0.1
        new_CR_weights = new_CR_weights / np.sum(new_CR_weights)
        self.CR_weights = momentum * self.CR_weights + (1 - momentum) * new_CR_weights
        self.CR_weights = self.CR_weights / np.sum(self.CR_weights)
        
        # Reset counters (keep some history)
        self.F_success_counts *= 0.5
        self.F_trial_counts *= 0.5
        self.CR_success_counts *= 0.5
        self.CR_trial_counts *= 0.5
    
    def get_statistics(self) -> dict:
        """
        Get current ensemble statistics.
        
        Returns
        -------
        stats : dict
            Dictionary containing weights, success rates, etc.
        """
        F_success_rates = np.zeros(len(self.F_values))
        for i in range(len(self.F_values)):
            if self.F_trial_counts[i] > 0:
                F_success_rates[i] = self.F_success_counts[i] / self.F_trial_counts[i]
        
        CR_success_rates = np.zeros(len(self.CR_values))
        for i in range(len(self.CR_values)):
            if self.CR_trial_counts[i] > 0:
                CR_success_rates[i] = self.CR_success_counts[i] / self.CR_trial_counts[i]
        
        return {
            'F_values': self.F_values,
            'F_weights': self.F_weights.tolist(),
            'F_success_rates': F_success_rates.tolist(),
            'CR_values': self.CR_values,
            'CR_weights': self.CR_weights.tolist(),
            'CR_success_rates': CR_success_rates.tolist(),
            'generation': self.generation,
        }
