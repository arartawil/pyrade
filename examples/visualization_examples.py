"""
Comprehensive examples for PyRADE visualization capabilities.

This script demonstrates all visualization types available in PyRADE:
- Convergence curves
- Fitness boxplots
- 2D/3D Pareto fronts
- Parameter heatmaps
- Parallel coordinate plots
- Contour landscape plots
- Hypervolume/IGD progress
"""

import numpy as np
import matplotlib.pyplot as plt
from pyrade import DifferentialEvolution
from pyrade.benchmarks import Sphere, Rastrigin, Rosenbrock, Ackley
from pyrade.visualization import (
    OptimizationVisualizer,
    calculate_hypervolume_2d,
    calculate_igd,
    is_pareto_efficient
)


def example_convergence_curves():
    """Example: Plot convergence curves comparing different strategies."""
    print("=" * 60)
    print("Example 1: Convergence Curves")
    print("=" * 60)
    
    # Run DE with different strategies
    results = {}
    strategies = ['rand/1/bin', 'best/1/bin', 'rand/2/bin']
    
    for strategy in strategies:
        de = DifferentialEvolution(
            obj_func=Sphere(),
            bounds=np.array([(-5.12, 5.12)] * 10),
            population_size=50,
            max_iterations=100,
            mutation_strategy=strategy.split('/')[0],
            F=0.8,
            CR=0.9
        )
        result = de.run()
        results[strategy] = result.get('fitness_history', [result['best_fitness']])
    
    # Create visualization
    viz = OptimizationVisualizer(figsize=(12, 6))
    fig = viz.plot_convergence_curve(
        history=results,
        title="DE Strategy Comparison on Sphere Function",
        ylabel="Best Fitness (log scale)",
        log_scale=True,
        save_path="convergence_comparison.png"
    )
    plt.show()
    print("✓ Convergence curves saved to 'convergence_comparison.png'\n")


def example_convergence_with_std():
    """Example: Convergence curve with standard deviation from multiple runs."""
    print("=" * 60)
    print("Example 2: Convergence with Standard Deviation")
    print("=" * 60)
    
    # Run multiple times
    n_runs = 10
    all_histories = []
    
    for i in range(n_runs):
        de = DifferentialEvolution(
            obj_func=Rastrigin(),
            bounds=np.array([(-5.12, 5.12)] * 5),
            population_size=30,
            max_iterations=50,
            F=0.8,
            CR=0.9,
            seed=i
        )
        result = de.run()
        all_histories.append(result.get('fitness_history', [result['best_fitness']]))
    
    viz = OptimizationVisualizer()
    fig = viz.plot_convergence_curve(
        history=all_histories,
        title="Rastrigin Function - 10 Independent Runs",
        show_std=True,
        log_scale=True,
        save_path="convergence_with_std.png"
    )
    plt.show()
    print("✓ Convergence with std saved to 'convergence_with_std.png'\n")


def example_fitness_boxplot():
    """Example: Boxplot comparing final fitness distributions."""
    print("=" * 60)
    print("Example 3: Fitness Distribution Boxplot")
    print("=" * 60)
    
    # Compare different mutation factors
    fitness_data = {}
    F_values = [0.3, 0.5, 0.7, 0.9]
    
    for F in F_values:
        fitness_list = []
        for run in range(20):
            de = DifferentialEvolution(
                obj_func=Rosenbrock(),
                bounds=np.array([(-5, 10)] * 8),
                population_size=40,
                max_iterations=100,
                F=F,
                CR=0.9,
                seed=run
            )
            result = de.run()
            fitness_list.append(result['best_fitness'])
        
        fitness_data[f'F={F}'] = fitness_list
    
    viz = OptimizationVisualizer(figsize=(10, 6))
    fig = viz.plot_fitness_boxplot(
        fitness_data=fitness_data,
        title="Effect of Mutation Factor F on Rosenbrock Function",
        ylabel="Final Best Fitness",
        save_path="fitness_boxplot.png"
    )
    plt.show()
    print("✓ Fitness boxplot saved to 'fitness_boxplot.png'\n")


def example_2d_pareto_front():
    """Example: 2D Pareto front scatter plot."""
    print("=" * 60)
    print("Example 4: 2D Pareto Front")
    print("=" * 60)
    
    # Generate multi-objective solutions (simulated)
    np.random.seed(42)
    n_solutions = 100
    
    # Simulate bi-objective optimization results
    # Front 1: x^2, (x-2)^2
    x = np.random.uniform(-1, 3, n_solutions)
    objectives = np.column_stack([x**2, (x - 2)**2])
    
    # Add some noise
    objectives += np.random.normal(0, 0.1, objectives.shape)
    
    # Identify Pareto front
    pareto_mask = is_pareto_efficient(objectives)
    
    viz = OptimizationVisualizer(figsize=(10, 8))
    fig = viz.plot_2d_pareto_front(
        objectives=objectives,
        pareto_front=pareto_mask,
        title="Bi-Objective Optimization Result",
        xlabel="Objective 1: f₁(x)",
        ylabel="Objective 2: f₂(x)",
        save_path="pareto_2d.png"
    )
    plt.show()
    print("✓ 2D Pareto front saved to 'pareto_2d.png'\n")


def example_3d_pareto_front():
    """Example: 3D Pareto front scatter plot."""
    print("=" * 60)
    print("Example 5: 3D Pareto Front")
    print("=" * 60)
    
    # Generate 3-objective solutions
    np.random.seed(42)
    n_solutions = 200
    
    # Simulate three-objective optimization
    theta = np.random.uniform(0, np.pi, n_solutions)
    phi = np.random.uniform(0, 2*np.pi, n_solutions)
    
    objectives = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])
    objectives = objectives**2 + np.random.normal(0, 0.05, objectives.shape)
    
    # Identify Pareto front
    pareto_mask = is_pareto_efficient(objectives)
    
    viz = OptimizationVisualizer(figsize=(12, 9))
    fig = viz.plot_3d_pareto_front(
        objectives=objectives,
        pareto_front=pareto_mask,
        title="Three-Objective Optimization Result",
        labels=("Objective 1", "Objective 2", "Objective 3"),
        save_path="pareto_3d.png"
    )
    plt.show()
    print("✓ 3D Pareto front saved to 'pareto_3d.png'\n")


def example_parameter_heatmap():
    """Example: Parameter value heatmap."""
    print("=" * 60)
    print("Example 6: Parameter Value Heatmap")
    print("=" * 60)
    
    # Run optimization and extract final population
    de = DifferentialEvolution(
        obj_func=Ackley(),
        bounds=np.array([(-32.768, 32.768)] * 6),
        population_size=50,
        max_iterations=100,
        F=0.8,
        CR=0.9,
        seed=42
    )
    result = de.run()
    
    # Get final population (simulated if not in result)
    if 'population' in result:
        population = result['population']
        fitness = result['population_fitness']
    else:
        # Simulate final population near optimum
        population = np.random.randn(50, 6) * 0.5
        fitness = np.array([Ackley()(ind) for ind in population])
    
    param_names = [f'x{i+1}' for i in range(6)]
    
    viz = OptimizationVisualizer(figsize=(12, 8))
    fig = viz.plot_parameter_heatmap(
        parameters=population,
        fitness=fitness,
        param_names=param_names,
        title="Parameter Distribution in Final Population (Ackley Function)",
        cmap='coolwarm',
        save_path="parameter_heatmap.png"
    )
    plt.show()
    print("✓ Parameter heatmap saved to 'parameter_heatmap.png'\n")


def example_parallel_coordinates():
    """Example: Parallel coordinate plot."""
    print("=" * 60)
    print("Example 7: Parallel Coordinate Plot")
    print("=" * 60)
    
    # Generate parameter space exploration data
    np.random.seed(42)
    n_samples = 100
    n_params = 5
    
    # Simulate parameters and fitness
    parameters = np.random.uniform(-5, 5, (n_samples, n_params))
    
    # Simulate fitness (Rosenbrock-like)
    fitness = np.sum((parameters[:, :-1]**2 - parameters[:, 1:])**2 + 
                     (1 - parameters[:, :-1])**2, axis=1)
    
    param_names = [f'Parameter {i+1}' for i in range(n_params)]
    
    viz = OptimizationVisualizer(figsize=(14, 7))
    fig = viz.plot_parallel_coordinates(
        parameters=parameters,
        fitness=fitness,
        param_names=param_names,
        normalize=True,
        title="Parameter Space Exploration - Parallel Coordinates",
        cmap='RdYlGn_r',
        save_path="parallel_coordinates.png"
    )
    plt.show()
    print("✓ Parallel coordinates saved to 'parallel_coordinates.png'\n")


def example_contour_landscape():
    """Example: Contour plot with optimization trajectory."""
    print("=" * 60)
    print("Example 8: Contour Landscape with Trajectory")
    print("=" * 60)
    
    # Define 2D Rastrigin
    rastrigin_2d = Rastrigin()
    bounds = np.array([[-5.12, 5.12], [-5.12, 5.12]])
    
    # Run optimization to get trajectory (simulated)
    np.random.seed(42)
    n_iterations = 20
    trajectory = []
    current = np.random.uniform(-5, 5, 2)
    
    for _ in range(n_iterations):
        trajectory.append(current.copy())
        # Simulate movement towards optimum
        direction = -current + np.random.randn(2) * 0.5
        current += direction * 0.3
        current = np.clip(current, bounds[:, 0], bounds[:, 1])
    
    trajectory = np.array(trajectory)
    
    viz = OptimizationVisualizer(figsize=(10, 8))
    fig = viz.plot_contour_landscape(
        benchmark_func=lambda x: rastrigin_2d(x),
        bounds=bounds,
        optimal_point=np.array([0, 0]),
        trajectory=trajectory,
        resolution=100,
        title="Rastrigin Function Landscape with Optimization Path",
        cmap='viridis',
        save_path="contour_landscape.png"
    )
    plt.show()
    print("✓ Contour landscape saved to 'contour_landscape.png'\n")


def example_hypervolume_progress():
    """Example: Hypervolume indicator progress."""
    print("=" * 60)
    print("Example 9: Hypervolume Progress")
    print("=" * 60)
    
    # Simulate multi-objective optimization
    np.random.seed(42)
    n_generations = 50
    hypervolume_history = []
    
    reference_point = np.array([10.0, 10.0])
    
    for gen in range(n_generations):
        # Simulate improving Pareto front
        n_points = 20
        t = np.linspace(0, 1, n_points)
        
        # Front improves over generations
        factor = 1.0 - 0.8 * (gen / n_generations)
        objectives = np.column_stack([
            t * factor * 5,
            (1 - t) * factor * 5
        ])
        
        # Calculate hypervolume
        hv = calculate_hypervolume_2d(objectives, reference_point)
        hypervolume_history.append(hv)
    
    # True reference hypervolume
    true_hv = 50.0  # Theoretical maximum
    
    viz = OptimizationVisualizer(figsize=(10, 6))
    fig = viz.plot_hypervolume_progress(
        hypervolume_history=hypervolume_history,
        title="Hypervolume Indicator Over Generations",
        reference_hv=true_hv,
        save_path="hypervolume_progress.png"
    )
    plt.show()
    print("✓ Hypervolume progress saved to 'hypervolume_progress.png'\n")


def example_igd_progress():
    """Example: IGD metric progress."""
    print("=" * 60)
    print("Example 10: IGD Progress")
    print("=" * 60)
    
    # Simulate IGD improvement
    np.random.seed(42)
    n_generations = 50
    
    # True Pareto front
    t = np.linspace(0, 1, 100)
    true_front = np.column_stack([t, 1 - t])
    
    igd_history = []
    
    for gen in range(n_generations):
        # Simulate approximation getting better
        n_points = 20
        noise_level = 2.0 * np.exp(-gen / 10)  # Decreasing noise
        
        t_approx = np.linspace(0, 1, n_points)
        obtained_front = np.column_stack([
            t_approx + np.random.randn(n_points) * noise_level * 0.1,
            1 - t_approx + np.random.randn(n_points) * noise_level * 0.1
        ])
        
        igd = calculate_igd(obtained_front, true_front)
        igd_history.append(igd)
    
    viz = OptimizationVisualizer(figsize=(10, 6))
    fig = viz.plot_igd_progress(
        igd_history=igd_history,
        title="Inverted Generational Distance Over Generations",
        log_scale=True,
        save_path="igd_progress.png"
    )
    plt.show()
    print("✓ IGD progress saved to 'igd_progress.png'\n")


def example_population_diversity():
    """Example: Population diversity over time."""
    print("=" * 60)
    print("Example 11: Population Diversity")
    print("=" * 60)
    
    # Simulate diversity metrics
    np.random.seed(42)
    n_generations = 100
    
    diversity_metrics = {
        'Std Dev': [],
        'Range': [],
        'Variance': []
    }
    
    for gen in range(n_generations):
        # Simulate diversity decreasing then stabilizing
        base_diversity = 5.0 * np.exp(-gen / 20) + 0.5
        
        diversity_metrics['Std Dev'].append(base_diversity + np.random.randn() * 0.1)
        diversity_metrics['Range'].append(base_diversity * 2 + np.random.randn() * 0.2)
        diversity_metrics['Variance'].append(base_diversity**2 + np.random.randn() * 0.5)
    
    viz = OptimizationVisualizer(figsize=(12, 6))
    fig = viz.plot_population_diversity(
        diversity_history=diversity_metrics,
        title="Population Diversity Metrics Over Generations",
        save_path="population_diversity.png"
    )
    plt.show()
    print("✓ Population diversity saved to 'population_diversity.png'\n")


def run_all_examples():
    """Run all visualization examples."""
    print("\n" + "=" * 60)
    print("PyRADE VISUALIZATION EXAMPLES")
    print("=" * 60 + "\n")
    
    examples = [
        example_convergence_curves,
        example_convergence_with_std,
        example_fitness_boxplot,
        example_2d_pareto_front,
        example_3d_pareto_front,
        example_parameter_heatmap,
        example_parallel_coordinates,
        example_contour_landscape,
        example_hypervolume_progress,
        example_igd_progress,
        example_population_diversity
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"✗ Example {i} failed: {e}\n")
    
    print("=" * 60)
    print("ALL EXAMPLES COMPLETED!")
    print("=" * 60)
    print("\nGenerated visualization files:")
    print("  - convergence_comparison.png")
    print("  - convergence_with_std.png")
    print("  - fitness_boxplot.png")
    print("  - pareto_2d.png")
    print("  - pareto_3d.png")
    print("  - parameter_heatmap.png")
    print("  - parallel_coordinates.png")
    print("  - contour_landscape.png")
    print("  - hypervolume_progress.png")
    print("  - igd_progress.png")
    print("  - population_diversity.png")


if __name__ == "__main__":
    # Run all examples
    run_all_examples()
    
    # Or run individual examples:
    # example_convergence_curves()
    # example_2d_pareto_front()
    # example_contour_landscape()
