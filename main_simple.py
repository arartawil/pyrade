"""
PyRADE Main Experiment Runner - Simplified Interface

Configure your experiment parameters below and run this file.

Usage:
    python main.py
"""

from pyrade import DErand1bin, DEbest1bin, DEcurrentToBest1bin, DErand2bin
from pyrade.benchmarks import sphere, rosenbrock, rastrigin, ackley, schwefel
from pyrade.runner import run_single, run_multiple, compare_algorithms


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Select Algorithm
ALGORITHM = DErand1bin
# ALGORITHM = DEbest1bin
# ALGORITHM = DEcurrentToBest1bin
# ALGORITHM = DErand2bin

# Select Benchmark Function
BENCHMARK_FUNC = sphere
# BENCHMARK_FUNC = rosenbrock
# BENCHMARK_FUNC = rastrigin
# BENCHMARK_FUNC = ackley
# BENCHMARK_FUNC = schwefel

# Problem Configuration
DIMENSIONS = 30
BOUNDS = (-100, 100)

# Algorithm Parameters
POPULATION_SIZE = 50
MAX_ITERATIONS = 1000
MUTATION_F = 0.8
CROSSOVER_CR = 0.9
RANDOM_SEED = 42

# Experiment Settings
NUM_RUNS = 10
VERBOSE = True
SAVE_RESULTS = True
SAVE_PLOTS = True
OUTPUT_DIR = "experimental"

# Visualization Configuration
# Options: 'all', 'basic', 'research', 'none'
# Or use dict for fine-grained control
VISUALIZATION_CONFIG = 'all'


# ============================================================================
# RUN EXPERIMENTS
# ============================================================================

if __name__ == "__main__":
    print("\nPyRADE Experiment Runner")
    print("=" * 80)
    print("\nSelect experiment type:")
    print("1. Single run")
    print("2. Multiple runs (with statistics)")
    print("3. Algorithm comparison")
    print("\n=" * 80)
    
    choice = input("\nEnter choice (1/2/3) or press Enter for single run: ").strip()
    
    if choice == "2":
        # Run multiple experiments
        run_multiple(
            algorithm=ALGORITHM,
            benchmark=BENCHMARK_FUNC,
            dimensions=DIMENSIONS,
            bounds=BOUNDS,
            pop_size=POPULATION_SIZE,
            max_iter=MAX_ITERATIONS,
            F=MUTATION_F,
            CR=CROSSOVER_CR,
            seed=RANDOM_SEED,
            num_runs=NUM_RUNS,
            save_results=SAVE_RESULTS,
            save_plots=SAVE_PLOTS,
            output_dir=OUTPUT_DIR,
            viz_config=VISUALIZATION_CONFIG
        )
    
    elif choice == "3":
        # Compare algorithms
        algorithms_to_compare = [
            DErand1bin,
            DEbest1bin,
            DEcurrentToBest1bin,
            DErand2bin,
        ]
        
        compare_algorithms(
            algorithms=algorithms_to_compare,
            benchmark=BENCHMARK_FUNC,
            dimensions=DIMENSIONS,
            bounds=BOUNDS,
            pop_size=POPULATION_SIZE,
            max_iter=MAX_ITERATIONS,
            F=MUTATION_F,
            CR=CROSSOVER_CR,
            seed=RANDOM_SEED,
            num_runs=NUM_RUNS,
            save_results=SAVE_RESULTS,
            save_plots=SAVE_PLOTS,
            output_dir=OUTPUT_DIR,
            viz_config=VISUALIZATION_CONFIG
        )
    
    else:
        # Run single experiment
        run_single(
            algorithm=ALGORITHM,
            benchmark=BENCHMARK_FUNC,
            dimensions=DIMENSIONS,
            bounds=BOUNDS,
            pop_size=POPULATION_SIZE,
            max_iter=MAX_ITERATIONS,
            F=MUTATION_F,
            CR=CROSSOVER_CR,
            seed=RANDOM_SEED,
            verbose=VERBOSE,
            save_results=SAVE_RESULTS,
            save_plots=SAVE_PLOTS,
            output_dir=OUTPUT_DIR,
            viz_config=VISUALIZATION_CONFIG
        )
