"""
Demonstration of Enhanced Operators in PyRADE v0.4.4.

This example shows how to use the new features:
1. Arithmetic and Three-point Crossover
2. Lévy Flight-based Mutation
3. Opposition-Based Learning (OBL) for initialization
4. Chaotic Maps for parameter control

Run: python examples/enhanced_operators_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pyrade.benchmarks import rastrigin, ackley, rosenbrock
from pyrade.operators import (
    ArithmeticCrossover,
    ThreePointCrossover,
    LevyFlightMutation,
    DErand1,
    BinomialCrossover
)
from pyrade.utils import (
    OppositionBasedLearning,
    apply_obl_initialization,
    LogisticMap,
    TentMap,
    SineMap,
    ChebyshevMap,
    ChaoticParameterController,
    create_chaotic_controller
)


def simple_de_optimization(
    objective_func,
    bounds,
    dim,
    pop_size,
    max_iter,
    mutation_strategy,
    crossover_strategy,
    use_obl=False,
    chaotic_controller=None,
    label="Standard"
):
    """Simple DE implementation for demonstration."""
    lower, upper = bounds
    lower_bounds = np.full(dim, lower)
    upper_bounds = np.full(dim, upper)
    
    # Initialize population
    if use_obl:
        obl = OppositionBasedLearning(opposition_type='simple')
        population, fitness = obl.initialize_population(
            objective_func, pop_size, bounds, dim
        )
    else:
        population = np.random.uniform(lower, upper, (pop_size, dim))
        fitness = np.array([objective_func(ind) for ind in population])
    
    history = []
    
    for generation in range(max_iter):
        # Get parameters
        if chaotic_controller:
            F, CR = chaotic_controller.get_parameters()
            chaotic_controller.update()
        else:
            F, CR = 0.8, 0.9
        
        # Get mutation factor for mutation strategy
        if hasattr(mutation_strategy, 'F'):
            mutation_strategy.F = F
        
        best_idx = np.argmin(fitness)
        target_indices = np.arange(pop_size)
        
        # Mutation
        mutants = mutation_strategy.apply(population, fitness, best_idx, target_indices)
        mutants = np.clip(mutants, lower_bounds, upper_bounds)
        
        # Crossover
        trials = crossover_strategy.apply(population, mutants)
        trials = np.clip(trials, lower_bounds, upper_bounds)
        
        # Selection
        trial_fitness = np.array([objective_func(trial) for trial in trials])
        improved = trial_fitness < fitness
        population[improved] = trials[improved]
        fitness[improved] = trial_fitness[improved]
        
        history.append(np.min(fitness))
        
        # OBL generation jumping (every 50 generations)
        if use_obl and generation % 50 == 0 and generation > 0:
            obl_temp = OppositionBasedLearning(opposition_type='quasi', jumping_rate=0.3)
            population, fitness = obl_temp.generation_jumping(
                population, fitness, objective_func, bounds
            )
    
    return history, np.min(fitness)


def demo_new_crossover_operators():
    """Demo 1: New Crossover Operators."""
    print("=" * 70)
    print("Demo 1: New Crossover Operators")
    print("=" * 70)
    
    def objective(x):
        return ackley(x)
    
    bounds = (-32.768, 32.768)
    dim = 20
    pop_size = 50
    max_iter = 200
    
    mutation = DErand1(F=0.8)
    
    crossovers = {
        'Binomial': BinomialCrossover(CR=0.9),
        'Arithmetic (α=0.5)': ArithmeticCrossover(alpha=0.5),
        'Arithmetic (adaptive)': ArithmeticCrossover(alpha=0.7, adaptive=True),
        'Three-Point': ThreePointCrossover()
    }
    
    results = {}
    print("\nTesting different crossover strategies on Ackley function...")
    
    for name, crossover in crossovers.items():
        print(f"  Running: {name}")
        history, final_fitness = simple_de_optimization(
            objective, bounds, dim, pop_size, max_iter,
            mutation, crossover, label=name
        )
        results[name] = history
        print(f"    Final fitness: {final_fitness:.6e}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for name, history in results.items():
        plt.semilogy(history, label=name, linewidth=2)
    
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (log scale)')
    plt.title('Crossover Operator Comparison on Ackley Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('crossover_comparison_demo.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved as 'crossover_comparison_demo.png'")
    plt.close()


def demo_levy_flight_mutation():
    """Demo 2: Lévy Flight Mutation."""
    print("\n" + "=" * 70)
    print("Demo 2: Lévy Flight Mutation")
    print("=" * 70)
    
    def objective(x):
        return rastrigin(x)
    
    bounds = (-5.12, 5.12)
    dim = 20
    pop_size = 50
    max_iter = 300
    
    crossover = BinomialCrossover(CR=0.9)
    
    mutations = {
        'Standard (DE/rand/1)': DErand1(F=0.8),
        'Lévy Flight (β=1.0)': LevyFlightMutation(beta=1.0, scale=0.01),
        'Lévy Flight (β=1.5)': LevyFlightMutation(beta=1.5, scale=0.01),
        'Lévy Flight (β=2.0)': LevyFlightMutation(beta=2.0, scale=0.01)
    }
    
    results = {}
    print("\nTesting Lévy flight mutation on Rastrigin function...")
    
    for name, mutation in mutations.items():
        print(f"  Running: {name}")
        history, final_fitness = simple_de_optimization(
            objective, bounds, dim, pop_size, max_iter,
            mutation, crossover, label=name
        )
        results[name] = history
        print(f"    Final fitness: {final_fitness:.6e}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for name, history in results.items():
        plt.semilogy(history, label=name, linewidth=2)
    
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (log scale)')
    plt.title('Lévy Flight Mutation vs Standard Mutation on Rastrigin')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('levy_mutation_demo.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved as 'levy_mutation_demo.png'")
    plt.close()


def demo_opposition_based_learning():
    """Demo 3: Opposition-Based Learning."""
    print("\n" + "=" * 70)
    print("Demo 3: Opposition-Based Learning (OBL)")
    print("=" * 70)
    
    def objective(x):
        return rosenbrock(x)
    
    bounds = (-5, 10)
    dim = 20
    pop_size = 50
    max_iter = 300
    
    mutation = DErand1(F=0.8)
    crossover = BinomialCrossover(CR=0.9)
    
    scenarios = {
        'Without OBL': (False, None),
        'OBL Init (simple)': (True, 'simple'),
        'OBL Init (quasi)': (True, 'quasi'),
        'OBL Init (generalized)': (True, 'generalized')
    }
    
    results = {}
    print("\nTesting Opposition-Based Learning on Rosenbrock function...")
    
    for name, (use_obl, obl_type) in scenarios.items():
        print(f"  Running: {name}")
        if use_obl:
            # Temporarily create OBL with specified type
            original_init = OppositionBasedLearning
        history, final_fitness = simple_de_optimization(
            objective, bounds, dim, pop_size, max_iter,
            mutation, crossover, use_obl=use_obl, label=name
        )
        results[name] = history
        print(f"    Final fitness: {final_fitness:.6e}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for name, history in results.items():
        plt.semilogy(history, label=name, linewidth=2)
    
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (log scale)')
    plt.title('Opposition-Based Learning Impact on Rosenbrock')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('obl_demo.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved as 'obl_demo.png'")
    plt.close()


def demo_chaotic_maps():
    """Demo 4: Chaotic Maps for Parameter Control."""
    print("\n" + "=" * 70)
    print("Demo 4: Chaotic Maps for Parameter Control")
    print("=" * 70)
    
    # First, visualize the chaotic sequences
    print("\nGenerating chaotic sequences...")
    
    maps = {
        'Logistic (r=4.0)': LogisticMap(r=4.0),
        'Tent': TentMap(),
        'Sine': SineMap(a=4.0),
        'Chebyshev (k=4)': ChebyshevMap(k=4)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (name, chaotic_map) in enumerate(maps.items()):
        sequence = chaotic_map.generate_sequence(200)
        axes[idx].plot(sequence, linewidth=1, alpha=0.7)
        axes[idx].set_title(f'{name} Sequence')
        axes[idx].set_xlabel('Iteration')
        axes[idx].set_ylabel('Value')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('chaotic_sequences_demo.png', dpi=150, bbox_inches='tight')
    print("✓ Plot saved as 'chaotic_sequences_demo.png'")
    plt.close()
    
    # Now test in DE optimization
    def objective(x):
        return ackley(x)
    
    bounds = (-32.768, 32.768)
    dim = 20
    pop_size = 50
    max_iter = 200
    
    mutation = DErand1(F=0.8)
    crossover = BinomialCrossover(CR=0.9)
    
    controllers = {
        'Fixed Parameters': None,
        'Logistic Map': create_chaotic_controller('logistic', F_range=(0.4, 0.9)),
        'Tent Map': create_chaotic_controller('tent', F_range=(0.4, 0.9)),
        'Sine Map': create_chaotic_controller('sine', F_range=(0.4, 0.9)),
        'Chebyshev Map': create_chaotic_controller('chebyshev', k=4, F_range=(0.4, 0.9))
    }
    
    results = {}
    print("\nTesting chaotic parameter control on Ackley function...")
    
    for name, controller in controllers.items():
        print(f"  Running: {name}")
        history, final_fitness = simple_de_optimization(
            objective, bounds, dim, pop_size, max_iter,
            mutation, crossover, chaotic_controller=controller, label=name
        )
        results[name] = history
        print(f"    Final fitness: {final_fitness:.6e}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for name, history in results.items():
        plt.semilogy(history, label=name, linewidth=2)
    
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (log scale)')
    plt.title('Chaotic Parameter Control vs Fixed Parameters')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('chaotic_control_demo.png', dpi=150, bbox_inches='tight')
    print("✓ Plot saved as 'chaotic_control_demo.png'")
    plt.close()


def demo_combined_features():
    """Demo 5: Combined Enhanced Features."""
    print("\n" + "=" * 70)
    print("Demo 5: Combined Enhanced Features")
    print("=" * 70)
    
    def objective(x):
        return rastrigin(x)
    
    bounds = (-5.12, 5.12)
    dim = 30
    pop_size = 60
    max_iter = 400
    
    scenarios = {
        'Standard DE': {
            'mutation': DErand1(F=0.8),
            'crossover': BinomialCrossover(CR=0.9),
            'obl': False,
            'chaotic': None
        },
        'Enhanced (Lévy + Arithmetic)': {
            'mutation': LevyFlightMutation(beta=1.5, scale=0.01),
            'crossover': ArithmeticCrossover(alpha=0.5, adaptive=True),
            'obl': False,
            'chaotic': None
        },
        'Enhanced + OBL': {
            'mutation': LevyFlightMutation(beta=1.5, scale=0.01),
            'crossover': ArithmeticCrossover(alpha=0.5, adaptive=True),
            'obl': True,
            'chaotic': None
        },
        'Full Enhancement': {
            'mutation': LevyFlightMutation(beta=1.5, scale=0.01),
            'crossover': ArithmeticCrossover(alpha=0.5, adaptive=True),
            'obl': True,
            'chaotic': create_chaotic_controller('logistic', F_range=(0.4, 0.9))
        }
    }
    
    results = {}
    print("\nTesting combined features on Rastrigin function (30D)...")
    
    for name, config in scenarios.items():
        print(f"  Running: {name}")
        history, final_fitness = simple_de_optimization(
            objective, bounds, dim, pop_size, max_iter,
            config['mutation'], config['crossover'],
            use_obl=config['obl'],
            chaotic_controller=config['chaotic'],
            label=name
        )
        results[name] = history
        print(f"    Final fitness: {final_fitness:.6e}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    for name, history in results.items():
        plt.semilogy(history, label=name, linewidth=2)
    
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (log scale)')
    plt.title('Combined Enhanced Features Performance (Rastrigin 30D)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('combined_features_demo.png', dpi=150, bbox_inches='tight')
    print("✓ Plot saved as 'combined_features_demo.png'")
    plt.close()


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("PyRADE v0.4.4 - Enhanced Operators Demonstration")
    print("=" * 70)
    print("\nThis demo showcases new enhanced operators:")
    print("  1. Arithmetic and Three-Point Crossover")
    print("  2. Lévy Flight-based Mutation")
    print("  3. Opposition-Based Learning (OBL)")
    print("  4. Chaotic Maps for Parameter Control")
    print("  5. Combined Features")
    print("=" * 70 + "\n")
    
    # Run all demos
    demo_new_crossover_operators()
    demo_levy_flight_mutation()
    demo_opposition_based_learning()
    demo_chaotic_maps()
    demo_combined_features()
    
    print("\n" + "=" * 70)
    print("All demonstrations completed successfully!")
    print("Check the generated PNG files for visualizations:")
    print("  - crossover_comparison_demo.png")
    print("  - levy_mutation_demo.png")
    print("  - obl_demo.png")
    print("  - chaotic_sequences_demo.png")
    print("  - chaotic_control_demo.png")
    print("  - combined_features_demo.png")
    print("=" * 70 + "\n")
