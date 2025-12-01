"""
PyRADE Main Experiment Runner

A unified interface for running experiments with any algorithm and benchmark function.
Configure your experiment by modifying the parameters below and run this file.

Usage:
    python main.py
"""

import numpy as np
from pyrade import (
    # Classic DE variants
    DErand1bin, DErand2bin, DEbest1bin, DEbest2bin,
    DEcurrentToBest1bin, DEcurrentToRand1bin, DERandToBest1bin,
    DErand1exp, DErand1EitherOrBin,
    # Legacy/Custom
    DifferentialEvolution
)
from pyrade.benchmarks import (
    # Simple functions (direct use)
    sphere, rosenbrock, rastrigin, ackley, schwefel,
    griewank, levy, michalewicz, zakharov, easom, styblinskitang,
    # Utilities
    get_benchmark, list_benchmarks
)
from pyrade.visualization import OptimizationVisualizer
from pyrade.experiments import ExperimentManager
import matplotlib.pyplot as plt


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# To see all available benchmarks:
# python -c "from pyrade.benchmarks import list_benchmarks; print(list_benchmarks())"

# Select Algorithm (uncomment one or add your own)
ALGORITHM = DErand1bin
# ALGORITHM = DEbest1bin
# ALGORITHM = DEcurrentToBest1bin
# ALGORITHM = DErand2bin
# ALGORITHM = DEbest2bin
# ALGORITHM = DErand1exp
# ALGORITHM = jDE  # Adaptive algorithm

# Select Benchmark Function 
# Method 1: Direct function import (simplest)
BENCHMARK_FUNC = sphere
# BENCHMARK_FUNC = rosenbrock
# BENCHMARK_FUNC = rastrigin
# BENCHMARK_FUNC = ackley
# BENCHMARK_FUNC = schwefel

# Method 2: Dynamic by name (more flexible)
# BENCHMARK_FUNC = get_benchmark('rastrigin')
# BENCHMARK_FUNC = get_benchmark('ackley')

# Method 3: CEC2017 competition functions (F1-F10 implemented)
# from pyrade.benchmarks import CEC2017Function
# BENCHMARK_FUNC = CEC2017Function(func_num=5, dimensions=30)  # F5: Rastrigin

# Problem Configuration
DIMENSIONS = 30
BOUNDS = (-100, 100)  # Can also be list of tuples: [(-10,10), (-5,5), ...]
                      # Or use: BENCHMARK_FUNC.get_bounds_array() for CEC2017

# Algorithm Parameters
POPULATION_SIZE = 50
MAX_ITERATIONS = 1000
MUTATION_F = 0.8
CROSSOVER_CR = 0.9
RANDOM_SEED = 42

# Experiment Settings
NUM_RUNS = 10  # Number of independent runs
VERBOSE = True
SAVE_RESULTS = True
OUTPUT_DIR = "results"

# Visualization Options
PLOT_CONVERGENCE = True
PLOT_POPULATION = False  # Only works for 2D problems
SAVE_PLOTS = True


# ============================================================================
# EXPERIMENT EXECUTION
# ============================================================================

def run_single_experiment():
    """Run a single optimization with the configured algorithm and function."""
    
    print("=" * 80)
    print("PyRADE - Single Experiment")
    print("=" * 80)
    print(f"Algorithm:  {ALGORITHM.__name__}")
    
    # Get function name (handle different types)
    func_name = getattr(BENCHMARK_FUNC, 'name', None) or \
                getattr(BENCHMARK_FUNC, '__name__', None) or \
                str(BENCHMARK_FUNC.__class__.__name__)
    print(f"Function:   {func_name}")
    print(f"Dimensions: {DIMENSIONS}")
    print(f"Bounds:     {BOUNDS}")
    print(f"Population: {POPULATION_SIZE}")
    print(f"Iterations: {MAX_ITERATIONS}")
    print("=" * 80)
    
    # Setup bounds
    if hasattr(BENCHMARK_FUNC, 'get_bounds_array'):
        # CEC2017 or class-based benchmark
        bounds = BENCHMARK_FUNC.get_bounds_array()
    elif isinstance(BOUNDS, tuple):
        bounds = [BOUNDS] * DIMENSIONS
    else:
        bounds = BOUNDS
    
    # Initialize algorithm
    optimizer = ALGORITHM(
        objective_func=BENCHMARK_FUNC,
        bounds=bounds,
        pop_size=POPULATION_SIZE,
        max_iter=MAX_ITERATIONS,
        F=MUTATION_F,
        CR=CROSSOVER_CR,
        seed=RANDOM_SEED,
        verbose=VERBOSE
    )
    
    # Run optimization
    result = optimizer.optimize()
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Best Fitness:     {result['best_fitness']:.6e}")
    print(f"Best Solution:    {result['best_solution'][:5]}..." if len(result['best_solution']) > 5 
          else f"Best Solution:    {result['best_solution']}")
    print(f"Iterations:       {result['n_iterations']}")
    print(f"Execution Time:   {result['time']:.3f}s")
    print("=" * 80)
    
    # Visualizations
    if PLOT_CONVERGENCE and 'history' in result:
        visualizer = OptimizationVisualizer()
        fig = visualizer.plot_convergence_curve(result['history'])
        plt.title(f"{ALGORITHM.__name__} on {func_name}")
        
        if SAVE_PLOTS:
            import os
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            filename = f"{OUTPUT_DIR}/{ALGORITHM.__name__}_{func_name}_convergence.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"\nConvergence plot saved to: {filename}")
        
        plt.show()
    
    return result


def run_multiple_experiments():
    """Run multiple independent experiments with statistical analysis."""
    
    # Get function name
    func_name = getattr(BENCHMARK_FUNC, 'name', None) or \
                getattr(BENCHMARK_FUNC, '__name__', None) or \
                str(BENCHMARK_FUNC.__class__.__name__)
    
    print("=" * 80)
    print("PyRADE - Multiple Runs Experiment")
    print("=" * 80)
    print(f"Algorithm:  {ALGORITHM.__name__}")
    print(f"Function:   {func_name}")
    print(f"Dimensions: {DIMENSIONS}")
    print(f"Runs:       {NUM_RUNS}")
    print("=" * 80)
    
    # Setup bounds
    if hasattr(BENCHMARK_FUNC, 'get_bounds_array'):
        bounds = BENCHMARK_FUNC.get_bounds_array()
    elif isinstance(BOUNDS, tuple):
        bounds = [BOUNDS] * DIMENSIONS
    else:
        bounds = BOUNDS
    
    # Storage for results
    all_best_fitness = []
    all_histories = []
    
    # Run multiple experiments
    for run in range(NUM_RUNS):
        print(f"\nRun {run + 1}/{NUM_RUNS}...", end=" ")
        
        # Initialize with different seed each run
        optimizer = ALGORITHM(
            objective_func=BENCHMARK_FUNC,
            bounds=bounds,
            pop_size=POPULATION_SIZE,
            max_iter=MAX_ITERATIONS,
            F=MUTATION_F,
            CR=CROSSOVER_CR,
            seed=RANDOM_SEED + run if RANDOM_SEED else None,
            verbose=False
        )
        
        result = optimizer.optimize()
        all_best_fitness.append(result['best_fitness'])
        if 'history' in result:
            all_histories.append(result['history'])
        
        print(f"Best: {result['best_fitness']:.6e}")
    
    # Statistical analysis
    best_fitness = np.array(all_best_fitness)
    print("\n" + "=" * 80)
    print("STATISTICAL RESULTS")
    print("=" * 80)
    print(f"Best:      {np.min(best_fitness):.6e}")
    print(f"Worst:     {np.max(best_fitness):.6e}")
    print(f"Mean:      {np.mean(best_fitness):.6e}")
    print(f"Median:    {np.median(best_fitness):.6e}")
    print(f"Std Dev:   {np.std(best_fitness):.6e}")
    print("=" * 80)
    
    # Plot convergence curves for all runs
    if PLOT_CONVERGENCE and all_histories:
        plt.figure(figsize=(10, 6))
        
        for i, history in enumerate(all_histories):
            plt.semilogy(history, alpha=0.3, color='blue')
        
        # Plot mean convergence
        histories_array = np.array(all_histories)
        mean_history = np.mean(histories_array, axis=0)
        plt.semilogy(mean_history, 'r-', linewidth=2, label='Mean')
        
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness (log scale)')
        plt.title(f'{ALGORITHM.__name__} on {func_name} ({NUM_RUNS} runs)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if SAVE_PLOTS:
            import os
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            filename = f"{OUTPUT_DIR}/{ALGORITHM.__name__}_{func_name}_multiple_runs.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"\nMultiple runs plot saved to: {filename}")
        
        plt.show()
    
    # Boxplot of final results
    plt.figure(figsize=(8, 6))
    plt.boxplot(best_fitness, vert=True)
    plt.ylabel('Best Fitness')
    plt.title(f'Distribution of Final Results\n{ALGORITHM.__name__} on {func_name}')
    plt.grid(True, alpha=0.3)
    
    if SAVE_PLOTS:
        import os
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename = f"{OUTPUT_DIR}/{ALGORITHM.__name__}_{func_name}_boxplot.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Boxplot saved to: {filename}")
    
    plt.show()
    
    return best_fitness, all_histories


def run_algorithm_comparison():
    """Compare multiple algorithms on the selected benchmark function."""
    
    # Get function name
    func_name = getattr(BENCHMARK_FUNC, 'name', None) or \
                getattr(BENCHMARK_FUNC, '__name__', None) or \
                str(BENCHMARK_FUNC.__class__.__name__)
    
    # Define algorithms to compare
    algorithms = [
        DErand1bin,
        DEbest1bin,
        DEcurrentToBest1bin,
        DErand2bin,
    ]
    
    print("=" * 80)
    print("PyRADE - Algorithm Comparison")
    print("=" * 80)
    print(f"Function:   {func_name}")
    print(f"Dimensions: {DIMENSIONS}")
    print(f"Algorithms: {len(algorithms)}")
    print(f"Runs/algo:  {NUM_RUNS}")
    print("=" * 80)
    
    # Setup bounds
    if hasattr(BENCHMARK_FUNC, 'get_bounds_array'):
        bounds = BENCHMARK_FUNC.get_bounds_array()
    elif isinstance(BOUNDS, tuple):
        bounds = [BOUNDS] * DIMENSIONS
    else:
        bounds = BOUNDS
    
    results = {}
    
    # Run each algorithm
    for algo in algorithms:
        print(f"\nTesting {algo.__name__}...")
        algo_results = []
        
        for run in range(NUM_RUNS):
            optimizer = algo(
                objective_func=BENCHMARK_FUNC,
                bounds=bounds,
                pop_size=POPULATION_SIZE,
                max_iter=MAX_ITERATIONS,
                F=MUTATION_F,
                CR=CROSSOVER_CR,
                seed=RANDOM_SEED + run if RANDOM_SEED else None,
                verbose=False
            )
            
            result = optimizer.optimize()
            algo_results.append(result['best_fitness'])
        
        results[algo.__name__] = np.array(algo_results)
        print(f"  Mean: {np.mean(algo_results):.6e}, Best: {np.min(algo_results):.6e}")
    
    # Statistical summary
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Algorithm':<25} {'Best':<15} {'Mean':<15} {'Std Dev':<15}")
    print("-" * 80)
    
    for name, fitness in results.items():
        print(f"{name:<25} {np.min(fitness):<15.6e} {np.mean(fitness):<15.6e} {np.std(fitness):<15.6e}")
    
    print("=" * 80)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    # Boxplot comparison
    plt.subplot(1, 2, 1)
    plt.boxplot(results.values(), labels=results.keys())
    plt.ylabel('Best Fitness')
    plt.title('Algorithm Comparison - Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Bar plot of means
    plt.subplot(1, 2, 2)
    means = [np.mean(v) for v in results.values()]
    stds = [np.std(v) for v in results.values()]
    x_pos = np.arange(len(results))
    plt.bar(x_pos, means, yerr=stds, capsize=5)
    plt.xticks(x_pos, results.keys(), rotation=45, ha='right')
    plt.ylabel('Mean Best Fitness')
    plt.title('Algorithm Comparison - Mean ± Std')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if SAVE_PLOTS:
        import os
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename = f"{OUTPUT_DIR}/algorithm_comparison_{func_name}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nComparison plot saved to: {filename}")
    
    plt.show()
    
    return results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\nPyRADE Experiment Runner")
    print("=" * 80)
    print("\nSelect experiment type:")
    print("1. Single run")
    print("2. Multiple runs (with statistics)")
    print("3. Algorithm comparison")
    print("\nOr edit main.py to configure and run directly.")
    print("=" * 80)
    
    choice = input("\nEnter choice (1/2/3) or press Enter for single run: ").strip()
    
    if choice == "2":
        run_multiple_experiments()
    elif choice == "3":
        run_algorithm_comparison()
    else:
        run_single_experiment()
