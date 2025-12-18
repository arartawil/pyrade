"""
PyRADE - Simple Experiment Interface

Just set your parameters and run! All the complex code is now in pyrade/runner.py

Quick Start:
    python main.py

Or import and use as functions:
    from main import run_experiment, compare_algorithms
"""

from pyrade import DErand1bin, DEbest1bin, DEcurrentToBest1bin
from pyrade.algorithms.adaptive import jDE, SaDE, JADE, SHADE, LSHADE, LSHADEEpSin, jSO, APSDE
from pyrade.benchmarks import sphere, rosenbrock, rastrigin, ackley, schwefel
from pyrade.benchmarks import get_benchmark, list_benchmarks
from pyrade.runner import run_experiment, compare_algorithms


# ============================================================================
# EXPERIMENT CONFIGURATION - Just set these parameters!
# ============================================================================

# Select Algorithm (uncomment one or add your own)
# Classic DE Algorithms:
ALGORITHM = DErand1bin
# ALGORITHM = DEbest1bin
# ALGORITHM = DEcurrentToBest1bin
# ALGORITHM = DErand2bin
# ALGORITHM = DEbest2bin
# ALGORITHM = DErand1exp

# Adaptive DE Algorithms (NEW in v0.4.0):
# ALGORITHM = jDE          # Self-adaptive F and CR
# ALGORITHM = SaDE         # Strategy pool adaptation
# ALGORITHM = JADE         # Archive-based adaptation
# ALGORITHM = SHADE        # Success-history adaptation
# ALGORITHM = LSHADE       # SHADE + population reduction
# ALGORITHM = LSHADEEpSin  # Ensemble + sinusoidal reduction
# ALGORITHM = jSO          # CEC 2020 winner
# ALGORITHM = APSDE        # Fitness-based adaptation

# Select Benchmark Function 
BENCHMARK_FUNC = sphere
# BENCHMARK_FUNC = rosenbrock
# BENCHMARK_FUNC = rastrigin
# BENCHMARK_FUNC = ackley
# BENCHMARK_FUNC = schwefel

# Or use dynamic selection:
# BENCHMARK_FUNC = get_benchmark('rastrigin')

# Or CEC2017 functions:
# from pyrade.benchmarks import CEC2017Function
# BENCHMARK_FUNC = CEC2017Function(func_num=5, dimensions=30)

# Or CEC2022 functions (NEW in v0.4.0):
# from pyrade.benchmarks import CEC2022
# BENCHMARK_FUNC = CEC2022(func_num=1, dim=10, data_dir='path/to/cec2022_data')
# NOTE: CEC2022 requires data files - see pyrade/benchmarks/CEC2022_README.md

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
NUM_RUNS = 10  # 1 for single run, >1 for statistics
OUTPUT_DIR = "experimental"

# Visualization
VISUALIZATION_PRESET = 'all'  # Options: 'all', 'basic', 'research', 'none'


# ============================================================================
# ADAPTIVE FEATURES (v0.4.2) - Optional Advanced Settings
# ============================================================================

# These features can be used to enhance optimization performance by dynamically
# adapting population size and parameters during the optimization process.

# Option 1: Adaptive Population Size
# -----------------------------------
# Automatically reduces or adjusts population size during optimization.
# Benefits: 30-50% faster convergence, reduced computational cost in later stages
#
# Example usage:
"""
from pyrade.utils import AdaptivePopulationSize

# Create adaptive population controller
aps = AdaptivePopulationSize(
    initial_size=100,           # Starting population size
    min_size=20,                # Minimum population (must be >= 4 for DE)
    strategy='lshade-like',     # Strategy: 'linear-reduction', 'lshade-like', 'success-based', 'diversity-based'
    reduction_rate=0.8          # How aggressively to reduce (0.0-1.0)
)

# Use in optimization loop:
for generation in range(max_iterations):
    # Update population size based on progress
    new_size = aps.update(
        generation=generation,
        max_generations=max_iterations,
        population=population,      # Current population array
        fitness=fitness,            # Current fitness values
        success_rate=success_rate   # Optional: improvement rate
    )
    
    # Check if resizing is needed
    should_resize, target_size = aps.should_resize(len(population))
    if should_resize:
        # Resize population (keeps best individuals when reducing)
        population, fitness = aps.resize_population(
            population, fitness, target_size
        )
    
    # ... rest of DE operations (mutation, crossover, selection)
"""

# Option 2: Parameter Ensemble
# -----------------------------
# Uses multiple F and CR parameter combinations simultaneously, with adaptive
# selection based on success history.
# Benefits: More robust across problems, automatic parameter tuning
#
# Example usage:
"""
from pyrade.utils import ParameterEnsemble

# Create parameter ensemble
ensemble = ParameterEnsemble(
    F_values=[0.4, 0.6, 0.8, 1.0],           # Pool of mutation factors
    CR_values=[0.1, 0.3, 0.5, 0.7, 0.9],     # Pool of crossover rates
    strategy='adaptive',                      # 'uniform', 'adaptive', or 'random'
    learning_period=25                        # Generations between weight updates
)

# Use in optimization loop:
for generation in range(max_iterations):
    # Sample parameters for entire population (each individual gets its own F and CR)
    F_array, CR_array, F_indices, CR_indices = ensemble.sample(pop_size)
    
    successful_indices = []
    
    # Apply DE operations with individual parameters
    for i in range(pop_size):
        # Mutation with individual F_array[i]
        mutant = population[r1] + F_array[i] * (population[r2] - population[r3])
        
        # Crossover with individual CR_array[i]
        trial = crossover(population[i], mutant, CR_array[i])
        
        # Selection
        if trial_fitness < fitness[i]:
            population[i] = trial
            fitness[i] = trial_fitness
            successful_indices.append(i)  # Track success
    
    # Update ensemble weights based on which parameters worked
    ensemble.update_success(
        np.array(successful_indices),
        F_indices,
        CR_indices
    )
    
    # Optionally check statistics every N generations
    if generation % 50 == 0:
        stats = ensemble.get_statistics()
        print(f"Gen {generation}: F weights = {stats['F_weights']}")
        print(f"Gen {generation}: CR weights = {stats['CR_weights']}")
"""

# Option 3: Combined Usage (Maximum Adaptivity)
# ----------------------------------------------
# Use both features together for best results
#
# Example usage:
"""
from pyrade.utils import AdaptivePopulationSize, ParameterEnsemble
import numpy as np

# Setup both mechanisms
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
population = np.random.uniform(lower, upper, (pop_size, dim))
fitness = np.array([objective(ind) for ind in population])

# Main optimization loop
for generation in range(max_iterations):
    current_pop_size = len(population)
    
    # 1. Sample parameters from ensemble
    F_array, CR_array, F_indices, CR_indices = ensemble.sample(current_pop_size)
    
    successful_indices = []
    
    # 2. DE operations with individual parameters
    for i in range(current_pop_size):
        # ... mutation with F_array[i]
        # ... crossover with CR_array[i]
        # ... selection and track successes
        pass
    
    # 3. Update ensemble
    ensemble.update_success(
        np.array(successful_indices),
        F_indices,
        CR_indices
    )
    
    # 4. Update population size
    success_rate = len(successful_indices) / current_pop_size
    new_size = aps.update(
        generation=generation,
        max_generations=max_iterations,
        population=population,
        fitness=fitness,
        success_rate=success_rate
    )
    
    # 5. Resize if needed
    should_resize, target_size = aps.should_resize(len(population))
    if should_resize:
        population, fitness = aps.resize_population(
            population, fitness, target_size
        )
        print(f"Generation {generation}: Resized to {target_size}")

# For a complete working example, run:
# python examples/adaptive_features_demo.py
"""


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
    print("\n" + "=" * 80)
    
    choice = input("\nEnter choice (1/2/3) or press Enter for single run: ").strip()
    
    if choice == "2":
        # Multiple runs
        stats, all_results = run_experiment(
            algorithm=ALGORITHM,
            benchmark=BENCHMARK_FUNC,
            dimensions=DIMENSIONS,
            bounds=BOUNDS,
            num_runs=NUM_RUNS,
            pop_size=POPULATION_SIZE,
            max_iter=MAX_ITERATIONS,
            F=MUTATION_F,
            CR=CROSSOVER_CR,
            seed=RANDOM_SEED,
            output_dir=OUTPUT_DIR,
            viz_preset=VISUALIZATION_PRESET
        )
        
    elif choice == "3":
        # Algorithm comparison
        algorithms_to_compare = [
            DErand1bin,
            DEbest1bin,
            DEcurrentToBest1bin,
            # Uncomment to compare with adaptive algorithms:
            # jDE,
            # JADE,
            # SHADE,
            # LSHADE,
        ]
        
        results = compare_algorithms(
            algorithms=algorithms_to_compare,
            benchmark=BENCHMARK_FUNC,
            dimensions=DIMENSIONS,
            bounds=BOUNDS,
            num_runs=NUM_RUNS,
            pop_size=POPULATION_SIZE,
            max_iter=MAX_ITERATIONS,
            F=MUTATION_F,
            CR=CROSSOVER_CR,
            seed=RANDOM_SEED,
            output_dir=OUTPUT_DIR,
            viz_preset=VISUALIZATION_PRESET
        )
        
    else:
        # Single run
        result = run_experiment(
            algorithm=ALGORITHM,
            benchmark=BENCHMARK_FUNC,
            dimensions=DIMENSIONS,
            bounds=BOUNDS,
            num_runs=1,
            pop_size=POPULATION_SIZE,
            max_iter=MAX_ITERATIONS,
            F=MUTATION_F,
            CR=CROSSOVER_CR,
            seed=RANDOM_SEED,
            output_dir=OUTPUT_DIR,
            viz_preset=VISUALIZATION_PRESET
        )
