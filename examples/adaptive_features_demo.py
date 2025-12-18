"""
Demonstration of Adaptive Population Size and Parameter Ensemble in PyRADE.

This example shows how to use the new v0.4.2 features:
1. Adaptive Population Size - dynamically adjusts population during optimization
2. Parameter Ensemble - mixes multiple F and CR parameter settings

Run: python examples/adaptive_features_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pyrade.algorithms.classic import DErand1bin
from pyrade.benchmarks import rosenbrock, rastrigin, ackley
from pyrade.utils import AdaptivePopulationSize, ParameterEnsemble


def simple_de_with_adaptive_population():
    """
    Example 1: Using Adaptive Population Size
    
    Demonstrates different population sizing strategies and their effects
    on optimization performance.
    """
    print("=" * 70)
    print("Example 1: Adaptive Population Size")
    print("=" * 70)
    
    # Define a test function
    def objective(x):
        return rastrigin(x)
    
    bounds = [(-5.12, 5.12)] * 20
    max_iter = 500
    
    # Test different strategies
    strategies = ['linear-reduction', 'lshade-like', 'success-based']
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        
        # Create adaptive population size controller
        aps = AdaptivePopulationSize(
            initial_size=100,
            min_size=20,
            strategy=strategy,
            reduction_rate=0.8
        )
        
        # Initialize population manually for demonstration
        dim = len(bounds)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        
        pop_size = aps.initial_size
        population = np.random.uniform(
            lower, upper, (pop_size, dim)
        )
        fitness = np.array([objective(ind) for ind in population])
        
        history = {'fitness': [], 'pop_size': []}
        
        # Manual optimization loop with adaptive population
        for gen in range(max_iter):
            # Update population size
            new_size = aps.update(
                generation=gen,
                max_generations=max_iter,
                population=population,
                fitness=fitness
            )
            
            # Resize if needed
            should_resize, target_size = aps.should_resize(len(population))
            if should_resize:
                population, fitness = aps.resize_population(
                    population, fitness, target_size
                )
                print(f"  Gen {gen}: Resized population to {target_size}")
            
            # Store history
            history['fitness'].append(np.min(fitness))
            history['pop_size'].append(len(population))
            
            # Simple DE mutation and crossover (simplified for demo)
            F, CR = 0.8, 0.9
            for i in range(len(population)):
                # Select random individuals
                indices = list(range(len(population)))
                indices.remove(i)
                r1, r2, r3 = np.random.choice(indices, 3, replace=False)
                
                # Mutation
                mutant = population[r1] + F * (population[r2] - population[r3])
                mutant = np.clip(mutant, lower, upper)
                
                # Crossover
                trial = population[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Selection
                trial_fitness = objective(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
        
        results[strategy] = history
        print(f"  Final best fitness: {np.min(fitness):.6e}")
        print(f"  Final population size: {len(population)}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    for strategy, history in results.items():
        ax1.semilogy(history['fitness'], label=strategy)
        ax2.plot(history['pop_size'], label=strategy)
    
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Fitness (log scale)')
    ax1.set_title('Convergence with Different Adaptive Population Strategies')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Population Size')
    ax2.set_title('Population Size Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('adaptive_population_demo.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved as 'adaptive_population_demo.png'")
    plt.close()


def parameter_ensemble_demo():
    """
    Example 2: Using Parameter Ensemble
    
    Demonstrates how to use multiple F and CR parameter combinations
    and adaptively select successful ones.
    """
    print("\n" + "=" * 70)
    print("Example 2: Parameter Ensemble")
    print("=" * 70)
    
    # Define test function
    def objective(x):
        return ackley(x)
    
    bounds = [(-32.768, 32.768)] * 15
    dim = len(bounds)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # Test different ensemble strategies
    strategies = ['uniform', 'adaptive']
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting ensemble strategy: {strategy}")
        
        # Create parameter ensemble
        ensemble = ParameterEnsemble(
            F_values=[0.4, 0.6, 0.8, 1.0],
            CR_values=[0.1, 0.3, 0.5, 0.7, 0.9],
            strategy=strategy,
            learning_period=25
        )
        
        # Initialize population
        pop_size = 50
        max_iter = 300
        population = np.random.uniform(lower, upper, (pop_size, dim))
        fitness = np.array([objective(ind) for ind in population])
        
        history = {'fitness': [], 'ensemble_stats': []}
        
        # Optimization loop with parameter ensemble
        for gen in range(max_iter):
            # Sample parameters from ensemble
            F_array, CR_array, F_indices, CR_indices = ensemble.sample(pop_size)
            
            successful_indices = []
            
            # Apply DE with ensemble parameters
            for i in range(pop_size):
                # Select random individuals
                indices = list(range(pop_size))
                indices.remove(i)
                r1, r2, r3 = np.random.choice(indices, 3, replace=False)
                
                # Mutation with individual F
                mutant = population[r1] + F_array[i] * (population[r2] - population[r3])
                mutant = np.clip(mutant, lower, upper)
                
                # Crossover with individual CR
                trial = population[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR_array[i] or j == j_rand:
                        trial[j] = mutant[j]
                
                # Selection
                trial_fitness = objective(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    successful_indices.append(i)
            
            # Update ensemble with success information
            ensemble.update_success(
                np.array(successful_indices),
                F_indices,
                CR_indices
            )
            
            # Store history
            history['fitness'].append(np.min(fitness))
            if gen % 50 == 0:
                stats = ensemble.get_statistics()
                history['ensemble_stats'].append(stats)
                print(f"  Gen {gen}: Best={np.min(fitness):.6e}, "
                      f"Success rate={len(successful_indices)/pop_size:.2%}")
        
        results[strategy] = history
        
        # Print final ensemble statistics
        final_stats = ensemble.get_statistics()
        print(f"\n  Final ensemble statistics:")
        print(f"  F weights: {[f'{w:.3f}' for w in final_stats['F_weights']]}")
        print(f"  CR weights: {[f'{w:.3f}' for w in final_stats['CR_weights']]}")
        print(f"  Final best fitness: {np.min(fitness):.6e}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Convergence comparison
    for strategy, history in results.items():
        axes[0, 0].semilogy(history['fitness'], label=strategy)
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Best Fitness (log scale)')
    axes[0, 0].set_title('Convergence: Uniform vs Adaptive Ensemble')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot parameter weight evolution for adaptive strategy
    if 'adaptive' in results:
        adaptive_stats = results['adaptive']['ensemble_stats']
        if len(adaptive_stats) > 0:
            generations = [i * 50 for i in range(len(adaptive_stats))]
            
            # F weights
            for i, F_val in enumerate(adaptive_stats[0]['F_values']):
                weights = [stats['F_weights'][i] for stats in adaptive_stats]
                axes[0, 1].plot(generations, weights, marker='o', label=f'F={F_val}')
            axes[0, 1].set_xlabel('Generation')
            axes[0, 1].set_ylabel('Weight')
            axes[0, 1].set_title('F Parameter Weight Evolution (Adaptive)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # CR weights
            for i, CR_val in enumerate(adaptive_stats[0]['CR_values']):
                weights = [stats['CR_weights'][i] for stats in adaptive_stats]
                axes[1, 0].plot(generations, weights, marker='s', label=f'CR={CR_val}')
            axes[1, 0].set_xlabel('Generation')
            axes[1, 0].set_ylabel('Weight')
            axes[1, 0].set_title('CR Parameter Weight Evolution (Adaptive)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
    
    # Final weight distribution
    if 'adaptive' in results:
        final_stats = results['adaptive']['ensemble_stats'][-1]
        
        axes[1, 1].bar(range(len(final_stats['F_values'])), 
                       final_stats['F_weights'],
                       alpha=0.7, label='F weights')
        axes[1, 1].set_xlabel('Parameter Index')
        axes[1, 1].set_ylabel('Final Weight')
        axes[1, 1].set_title('Final Parameter Weights (Adaptive)')
        axes[1, 1].set_xticks(range(len(final_stats['F_values'])))
        axes[1, 1].set_xticklabels([f"F={v}" for v in final_stats['F_values']], rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parameter_ensemble_demo.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved as 'parameter_ensemble_demo.png'")
    plt.close()


def combined_features_demo():
    """
    Example 3: Combining Adaptive Population and Parameter Ensemble
    
    Shows how to use both features together for maximum adaptivity.
    """
    print("\n" + "=" * 70)
    print("Example 3: Combined Adaptive Features")
    print("=" * 70)
    
    def objective(x):
        return rosenbrock(x)
    
    bounds = [(-5, 10)] * 30
    dim = len(bounds)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    
    # Setup both adaptive mechanisms
    aps = AdaptivePopulationSize(
        initial_size=120,
        min_size=30,
        strategy='lshade-like'
    )
    
    ensemble = ParameterEnsemble(
        F_values=[0.5, 0.7, 0.9],
        CR_values=[0.1, 0.5, 0.9],
        strategy='adaptive',
        learning_period=20
    )
    
    # Initialize
    pop_size = aps.initial_size
    max_iter = 400
    population = np.random.uniform(lower, upper, (pop_size, dim))
    fitness = np.array([objective(ind) for ind in population])
    
    history = {
        'fitness': [],
        'pop_size': [],
        'success_rate': []
    }
    
    print("\nRunning optimization with combined adaptive features...")
    
    for gen in range(max_iter):
        current_pop_size = len(population)
        
        # Sample parameters from ensemble
        F_array, CR_array, F_indices, CR_indices = ensemble.sample(current_pop_size)
        
        successful_indices = []
        
        # DE operations
        for i in range(current_pop_size):
            indices = list(range(current_pop_size))
            indices.remove(i)
            r1, r2, r3 = np.random.choice(indices, 3, replace=False)
            
            mutant = population[r1] + F_array[i] * (population[r2] - population[r3])
            mutant = np.clip(mutant, lower, upper)
            
            trial = population[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.rand() < CR_array[i] or j == j_rand:
                    trial[j] = mutant[j]
            
            trial_fitness = objective(trial)
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                successful_indices.append(i)
        
        # Update ensemble
        ensemble.update_success(
            np.array(successful_indices),
            F_indices,
            CR_indices
        )
        
        # Calculate success rate
        success_rate = len(successful_indices) / current_pop_size
        
        # Update population size
        new_size = aps.update(
            generation=gen,
            max_generations=max_iter,
            population=population,
            fitness=fitness,
            success_rate=success_rate
        )
        
        # Resize if needed
        should_resize, target_size = aps.should_resize(len(population))
        if should_resize:
            population, fitness = aps.resize_population(
                population, fitness, target_size
            )
        
        # Store history
        history['fitness'].append(np.min(fitness))
        history['pop_size'].append(len(population))
        history['success_rate'].append(success_rate)
        
        if gen % 50 == 0:
            print(f"  Gen {gen}: Best={np.min(fitness):.6e}, "
                  f"PopSize={len(population)}, Success={success_rate:.2%}")
    
    print(f"\n✓ Optimization complete!")
    print(f"  Final best fitness: {np.min(fitness):.6e}")
    print(f"  Final population size: {len(population)}")
    
    # Plot combined results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    axes[0].semilogy(history['fitness'], color='blue', linewidth=2)
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Best Fitness (log scale)')
    axes[0].set_title('Convergence with Combined Adaptive Features')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['pop_size'], color='green', linewidth=2)
    axes[1].set_xlabel('Generation')
    axes[1].set_ylabel('Population Size')
    axes[1].set_title('Adaptive Population Size Over Time')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(history['success_rate'], color='orange', linewidth=2)
    axes[2].set_xlabel('Generation')
    axes[2].set_ylabel('Success Rate')
    axes[2].set_title('Improvement Success Rate')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('combined_adaptive_demo.png', dpi=150, bbox_inches='tight')
    print("✓ Plot saved as 'combined_adaptive_demo.png'")
    plt.close()


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("PyRADE v0.4.2 - Adaptive Features Demonstration")
    print("=" * 70)
    print("\nThis demo showcases two new adaptive mechanisms:")
    print("  1. Adaptive Population Size")
    print("  2. Parameter Ensemble")
    print("=" * 70)
    
    # Run demos
    simple_de_with_adaptive_population()
    parameter_ensemble_demo()
    combined_features_demo()
    
    print("\n" + "=" * 70)
    print("All demonstrations completed successfully!")
    print("Check the generated PNG files for visualizations.")
    print("=" * 70 + "\n")
