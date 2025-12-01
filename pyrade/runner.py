"""
PyRADE Experiment Runner

High-level functions for running optimization experiments with automatic
visualization and result saving.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import csv


def run_single(algorithm, benchmark, dimensions=30, bounds=(-100, 100),
               pop_size=50, max_iter=1000, F=0.8, CR=0.9, seed=42,
               verbose=True, save_results=True, save_plots=True,
               output_dir="experimental", viz_config='all'):
    """
    Run a single optimization experiment.
    
    Parameters
    ----------
    algorithm : class
        DE algorithm class (e.g., DErand1bin, DEbest1bin)
    benchmark : callable
        Benchmark function to optimize
    dimensions : int
        Problem dimensionality
    bounds : tuple or list
        Search space bounds
    pop_size : int
        Population size
    max_iter : int
        Maximum iterations
    F : float
        Mutation factor
    CR : float
        Crossover rate
    seed : int
        Random seed
    verbose : bool
        Print progress during optimization
    save_results : bool
        Save results to CSV
    save_plots : bool
        Save visualization plots
    output_dir : str
        Base output directory
    viz_config : str or dict
        Visualization configuration ('all', 'basic', 'research', 'none' or dict)
    
    Returns
    -------
    dict
        Optimization results with best_fitness, best_solution, time, etc.
    """
    from pyrade.visualization import OptimizationVisualizer
    
    # Create experiment folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_folder = os.path.join(output_dir, timestamp)
    os.makedirs(exp_folder, exist_ok=True)
    
    # Get function name
    func_name = getattr(benchmark, 'name', None) or \
                getattr(benchmark, '__name__', None) or \
                str(benchmark.__class__.__name__)
    
    print("=" * 80)
    print("PyRADE - Single Experiment")
    print("=" * 80)
    print(f"Experiment:  {timestamp}")
    print(f"Output Dir:  {exp_folder}")
    print(f"Algorithm:   {algorithm.__name__}")
    print(f"Function:    {func_name}")
    print(f"Dimensions:  {dimensions}")
    print(f"Bounds:      {bounds}")
    print(f"Population:  {pop_size}")
    print(f"Iterations:  {max_iter}")
    print("=" * 80)
    
    # Setup bounds
    if hasattr(benchmark, 'get_bounds_array'):
        bounds_array = benchmark.get_bounds_array()
    elif isinstance(bounds, tuple):
        bounds_array = [bounds] * dimensions
    else:
        bounds_array = bounds
    
    # Initialize and run
    optimizer = algorithm(
        objective_func=benchmark,
        bounds=bounds_array,
        pop_size=pop_size,
        max_iter=max_iter,
        F=F,
        CR=CR,
        seed=seed,
        verbose=verbose
    )
    
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
    
    # Save results
    if save_results:
        _save_single_results(exp_folder, result, algorithm, func_name, 
                           dimensions, pop_size, max_iter, F, CR)
    
    # Generate visualizations
    if save_plots:
        plots_count = _generate_single_visualizations(
            exp_folder, result, algorithm, func_name, dimensions, 
            bounds, benchmark, viz_config
        )
    
    return result


def run_multiple(algorithm, benchmark, dimensions=30, bounds=(-100, 100),
                 pop_size=50, max_iter=1000, F=0.8, CR=0.9, seed=42,
                 num_runs=10, save_results=True, save_plots=True,
                 output_dir="experimental", viz_config='all'):
    """
    Run multiple independent optimization experiments with statistics.
    
    Parameters
    ----------
    algorithm : class
        DE algorithm class
    benchmark : callable
        Benchmark function
    dimensions : int
        Problem dimensionality
    bounds : tuple or list
        Search space bounds
    pop_size : int
        Population size
    max_iter : int
        Maximum iterations
    F : float
        Mutation factor
    CR : float
        Crossover rate
    seed : int
        Base random seed (incremented for each run)
    num_runs : int
        Number of independent runs
    save_results : bool
        Save results to CSV
    save_plots : bool
        Save visualization plots
    output_dir : str
        Base output directory
    viz_config : str or dict
        Visualization configuration
    
    Returns
    -------
    tuple
        (fitness_array, histories_list) with all run results
    """
    from pyrade.visualization import OptimizationVisualizer
    
    # Create experiment folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_folder = os.path.join(output_dir, timestamp)
    os.makedirs(exp_folder, exist_ok=True)
    
    func_name = getattr(benchmark, 'name', None) or \
                getattr(benchmark, '__name__', None) or \
                str(benchmark.__class__.__name__)
    
    print("=" * 80)
    print("PyRADE - Multiple Runs Experiment")
    print("=" * 80)
    print(f"Experiment:  {timestamp}")
    print(f"Output Dir:  {exp_folder}")
    print(f"Algorithm:   {algorithm.__name__}")
    print(f"Function:    {func_name}")
    print(f"Dimensions:  {dimensions}")
    print(f"Runs:        {num_runs}")
    print("=" * 80)
    
    # Setup bounds
    if hasattr(benchmark, 'get_bounds_array'):
        bounds_array = benchmark.get_bounds_array()
    elif isinstance(bounds, tuple):
        bounds_array = [bounds] * dimensions
    else:
        bounds_array = bounds
    
    # Run experiments
    all_fitness = []
    all_histories = []
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}...", end=" ")
        
        optimizer = algorithm(
            objective_func=benchmark,
            bounds=bounds_array,
            pop_size=pop_size,
            max_iter=max_iter,
            F=F,
            CR=CR,
            seed=seed + run if seed else None,
            verbose=False
        )
        
        result = optimizer.optimize()
        all_fitness.append(result['best_fitness'])
        if 'history' in result:
            all_histories.append(result['history'])
        
        print(f"Best: {result['best_fitness']:.6e}")
    
    # Statistics
    fitness = np.array(all_fitness)
    print("\n" + "=" * 80)
    print("STATISTICAL RESULTS")
    print("=" * 80)
    print(f"Best:      {np.min(fitness):.6e}")
    print(f"Worst:     {np.max(fitness):.6e}")
    print(f"Mean:      {np.mean(fitness):.6e}")
    print(f"Median:    {np.median(fitness):.6e}")
    print(f"Std Dev:   {np.std(fitness):.6e}")
    print("=" * 80)
    
    # Save results
    if save_results:
        _save_multiple_results(exp_folder, fitness, algorithm, func_name,
                              dimensions, num_runs)
    
    # Visualizations
    if save_plots:
        plots_count = _generate_multiple_visualizations(
            exp_folder, fitness, all_histories, algorithm, func_name,
            num_runs, viz_config
        )
    
    return fitness, all_histories


def compare_algorithms(algorithms, benchmark, dimensions=30, bounds=(-100, 100),
                       pop_size=50, max_iter=1000, F=0.8, CR=0.9, seed=42,
                       num_runs=10, save_results=True, save_plots=True,
                       output_dir="experimental", viz_config='all'):
    """
    Compare multiple algorithms on a benchmark function.
    
    Parameters
    ----------
    algorithms : list
        List of DE algorithm classes to compare
    benchmark : callable
        Benchmark function
    dimensions : int
        Problem dimensionality
    bounds : tuple or list
        Search space bounds
    pop_size : int
        Population size
    max_iter : int
        Maximum iterations
    F : float
        Mutation factor
    CR : float
        Crossover rate
    seed : int
        Base random seed
    num_runs : int
        Number of runs per algorithm
    save_results : bool
        Save results to CSV
    save_plots : bool
        Save visualization plots
    output_dir : str
        Base output directory
    viz_config : str or dict
        Visualization configuration
    
    Returns
    -------
    dict
        Results for each algorithm {algo_name: fitness_array}
    """
    from pyrade.visualization import OptimizationVisualizer
    
    # Create experiment folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_folder = os.path.join(output_dir, timestamp)
    os.makedirs(exp_folder, exist_ok=True)
    
    func_name = getattr(benchmark, 'name', None) or \
                getattr(benchmark, '__name__', None) or \
                str(benchmark.__class__.__name__)
    
    print("=" * 80)
    print("PyRADE - Algorithm Comparison")
    print("=" * 80)
    print(f"Experiment:  {timestamp}")
    print(f"Output Dir:  {exp_folder}")
    print(f"Function:    {func_name}")
    print(f"Dimensions:  {dimensions}")
    print(f"Algorithms:  {len(algorithms)}")
    print(f"Runs/algo:   {num_runs}")
    print("=" * 80)
    
    # Setup bounds
    if hasattr(benchmark, 'get_bounds_array'):
        bounds_array = benchmark.get_bounds_array()
    elif isinstance(bounds, tuple):
        bounds_array = [bounds] * dimensions
    else:
        bounds_array = bounds
    
    results = {}
    histories = {}
    
    # Run each algorithm
    for algo in algorithms:
        print(f"\nTesting {algo.__name__}...")
        algo_fitness = []
        algo_histories = []
        
        for run in range(num_runs):
            optimizer = algo(
                objective_func=benchmark,
                bounds=bounds_array,
                pop_size=pop_size,
                max_iter=max_iter,
                F=F,
                CR=CR,
                seed=seed + run if seed else None,
                verbose=False
            )
            
            result = optimizer.optimize()
            algo_fitness.append(result['best_fitness'])
            if 'history' in result and isinstance(result['history'], dict):
                algo_histories.append(result['history']['fitness'])
        
        results[algo.__name__] = np.array(algo_fitness)
        if algo_histories:
            histories[algo.__name__] = np.array(algo_histories)
        print(f"  Mean: {np.mean(algo_fitness):.6e}, Best: {np.min(algo_fitness):.6e}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Algorithm':<25} {'Best':<15} {'Mean':<15} {'Std Dev':<15}")
    print("-" * 80)
    for name, fitness in results.items():
        print(f"{name:<25} {np.min(fitness):<15.6e} {np.mean(fitness):<15.6e} {np.std(fitness):<15.6e}")
    print("=" * 80)
    
    # Save results
    if save_results:
        _save_comparison_results(exp_folder, results, func_name, dimensions,
                                num_runs, pop_size, max_iter)
    
    # Visualizations
    if save_plots:
        plots_count = _generate_comparison_visualizations(
            exp_folder, results, histories, func_name, viz_config
        )
    
    return results


# ============================================================================
# Helper Functions
# ============================================================================

def _parse_viz_config(viz_config):
    """Parse visualization configuration."""
    if isinstance(viz_config, str):
        presets = {
            'all': {k: True for k in [
                'convergence_curve', 'fitness_boxplot', 'parameter_heatmap',
                'parallel_coordinates', 'population_diversity', 'contour_landscape',
                'pareto_front_2d', 'pareto_front_3d', 'hypervolume_progress', 'igd_progress'
            ]},
            'basic': {
                'convergence_curve': True, 'fitness_boxplot': True,
                'parameter_heatmap': False, 'parallel_coordinates': False,
                'population_diversity': False, 'contour_landscape': False,
                'pareto_front_2d': False, 'pareto_front_3d': False,
                'hypervolume_progress': False, 'igd_progress': False
            },
            'research': {
                'convergence_curve': True, 'fitness_boxplot': True,
                'parameter_heatmap': True, 'parallel_coordinates': True,
                'population_diversity': True, 'contour_landscape': False,
                'pareto_front_2d': False, 'pareto_front_3d': False,
                'hypervolume_progress': False, 'igd_progress': False
            },
            'none': {k: False for k in [
                'convergence_curve', 'fitness_boxplot', 'parameter_heatmap',
                'parallel_coordinates', 'population_diversity', 'contour_landscape',
                'pareto_front_2d', 'pareto_front_3d', 'hypervolume_progress', 'igd_progress'
            ]}
        }
        return presets.get(viz_config.lower(), presets['basic'])
    return viz_config


def _save_single_results(folder, result, algo, func_name, dims, pop, iters, F, CR):
    """Save single run results to CSV."""
    csv_file = f"{folder}/single_run_results.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Algorithm', algo.__name__])
        writer.writerow(['Function', func_name])
        writer.writerow(['Dimensions', dims])
        writer.writerow(['Population Size', pop])
        writer.writerow(['Max Iterations', iters])
        writer.writerow(['Mutation F', F])
        writer.writerow(['Crossover CR', CR])
        writer.writerow(['Best Fitness', result['best_fitness']])
        writer.writerow(['Execution Time (s)', result['time']])
        writer.writerow([''])
        writer.writerow(['Best Solution'])
        for i, val in enumerate(result['best_solution']):
            writer.writerow([f'x[{i}]', val])
    print(f"\nResults saved to: {csv_file}")
    
    # Save history
    if 'history' in result and isinstance(result['history'], dict):
        hist_file = f"{folder}/convergence_history.csv"
        with open(hist_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Iteration', 'Best Fitness', 'Time'])
            for i, (fit, t) in enumerate(zip(result['history']['fitness'], 
                                               result['history']['time'])):
                writer.writerow([i, fit, t])
        print(f"Convergence history saved to: {hist_file}")


def _save_multiple_results(folder, fitness, algo, func_name, dims, runs):
    """Save multiple runs results to CSV."""
    stats_file = f"{folder}/multiple_runs_statistics.csv"
    with open(stats_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Algorithm', algo.__name__])
        writer.writerow(['Function', func_name])
        writer.writerow(['Dimensions', dims])
        writer.writerow(['Number of Runs', runs])
        writer.writerow([''])
        writer.writerow(['Statistical Results', ''])
        writer.writerow(['Best', np.min(fitness)])
        writer.writerow(['Worst', np.max(fitness)])
        writer.writerow(['Mean', np.mean(fitness)])
        writer.writerow(['Median', np.median(fitness)])
        writer.writerow(['Std Dev', np.std(fitness)])
    print(f"\nStatistics saved to: {stats_file}")
    
    runs_file = f"{folder}/all_runs_results.csv"
    with open(runs_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Run', 'Best Fitness'])
        for i, fit in enumerate(fitness, 1):
            writer.writerow([i, fit])
    print(f"All runs results saved to: {runs_file}")


def _save_comparison_results(folder, results, func_name, dims, runs, pop, iters):
    """Save algorithm comparison results to CSV."""
    summary_file = f"{folder}/algorithm_comparison_summary.csv"
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Experiment Info', ''])
        writer.writerow(['Function', func_name])
        writer.writerow(['Dimensions', dims])
        writer.writerow(['Runs per Algorithm', runs])
        writer.writerow(['Population Size', pop])
        writer.writerow(['Max Iterations', iters])
        writer.writerow([''])
        writer.writerow(['Algorithm', 'Best', 'Mean', 'Std Dev', 'Worst'])
        for name, fitness in results.items():
            writer.writerow([name, np.min(fitness), np.mean(fitness), 
                           np.std(fitness), np.max(fitness)])
    print(f"\nComparison summary saved to: {summary_file}")
    
    detailed_file = f"{folder}/algorithm_comparison_detailed.csv"
    with open(detailed_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Run'] + list(results.keys())
        writer.writerow(header)
        for run in range(len(next(iter(results.values())))):
            row = [run + 1] + [fitness[run] for fitness in results.values()]
            writer.writerow(row)
    print(f"Detailed results saved to: {detailed_file}")


def _generate_single_visualizations(folder, result, algo, func_name, dims, bounds, benchmark, viz_config):
    """Generate visualizations for single run."""
    from pyrade.visualization import OptimizationVisualizer
    
    viz_config = _parse_viz_config(viz_config)
    visualizer = OptimizationVisualizer()
    
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    plots_generated = 0
    
    # Import the visualization generation code here
    # (Same as in main.py but cleaner)
    
    # 1. Convergence curve
    if viz_config.get('convergence_curve') and 'history' in result:
        try:
            fig = visualizer.plot_convergence_curve(result['history'])
            plt.title(f"{algo.__name__} on {func_name}")
            filename = f"{folder}/convergence.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✓ Convergence curve: {filename}")
            plots_generated += 1
            plt.close()
        except Exception as e:
            print(f"✗ Convergence curve failed: {e}")
    
    # 2. Solution parameters
    if viz_config.get('parameter_heatmap') and 'best_solution' in result and len(result['best_solution']) <= 50:
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            dims_range = np.arange(len(result['best_solution']))
            ax.bar(dims_range, result['best_solution'], alpha=0.7, edgecolor='black')
            ax.set_xlabel('Dimension')
            ax.set_ylabel('Value')
            ax.set_title(f'Best Solution Parameters - {func_name}')
            ax.grid(True, alpha=0.3)
            filename = f"{folder}/solution_parameters.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✓ Solution parameters: {filename}")
            plots_generated += 1
            plt.close()
        except Exception as e:
            print(f"✗ Solution parameters failed: {e}")
    
    # 3. Convergence analysis
    if viz_config.get('convergence_curve') and 'history' in result:
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            fitness_history = result['history']['fitness']
            iterations = range(len(fitness_history))
            
            ax1.semilogy(iterations, fitness_history, 'b-', linewidth=2)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Best Fitness (log scale)')
            ax1.set_title(f'Convergence (Log Scale) - {func_name}')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(iterations, fitness_history, 'g-', linewidth=2)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Best Fitness')
            ax2.set_title(f'Convergence (Linear Scale) - {func_name}')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            filename = f"{folder}/convergence_analysis.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✓ Convergence analysis: {filename}")
            plots_generated += 1
            plt.close()
        except Exception as e:
            print(f"✗ Convergence analysis failed: {e}")
    
    # 4. Improvement rate
    if viz_config.get('population_diversity') and 'history' in result:
        try:
            fitness_history = np.array(result['history']['fitness'])
            improvements = np.diff(fitness_history)
            improvement_rate = np.abs(improvements) / (fitness_history[:-1] + 1e-10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(len(improvement_rate)), improvement_rate, 'r-', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Improvement Rate')
            ax.set_title(f'Fitness Improvement Rate - {func_name}')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            filename = f"{folder}/improvement_rate.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✓ Improvement rate: {filename}")
            plots_generated += 1
            plt.close()
        except Exception as e:
            print(f"✗ Improvement rate failed: {e}")
    
    # 5. 2D landscape (if applicable)
    if viz_config.get('contour_landscape') and dims == 2:
        try:
            x_range = np.linspace(bounds[0] if isinstance(bounds, tuple) else bounds[0][0],
                                 bounds[1] if isinstance(bounds, tuple) else bounds[0][1], 100)
            y_range = np.linspace(bounds[0] if isinstance(bounds, tuple) else bounds[1][0] if isinstance(bounds, list) else bounds[0],
                                 bounds[1] if isinstance(bounds, tuple) else bounds[1][1] if isinstance(bounds, list) else bounds[1], 100)
            X, Y = np.meshgrid(x_range, y_range)
            Z = np.zeros_like(X)
            
            for i in range(len(x_range)):
                for j in range(len(y_range)):
                    Z[j, i] = benchmark(np.array([X[j, i], Y[j, i]]))
            
            fig, ax = plt.subplots(figsize=(10, 8))
            contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
            plt.colorbar(contour, ax=ax, label='Fitness')
            ax.plot(result['best_solution'][0], result['best_solution'][1], 
                   'r*', markersize=20, label='Best Solution')
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_title(f'Fitness Landscape - {func_name}')
            ax.legend()
            filename = f"{folder}/fitness_landscape_2d.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✓ Fitness landscape: {filename}")
            plots_generated += 1
            plt.close()
        except Exception as e:
            print(f"✗ Fitness landscape failed: {e}")
    
    print(f"\nTotal plots generated: {plots_generated}")
    print("=" * 80)
    return plots_generated


def _generate_multiple_visualizations(folder, fitness, histories, algo, func_name, num_runs, viz_config):
    """Generate visualizations for multiple runs."""
    viz_config = _parse_viz_config(viz_config)
    
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    plots_generated = 0
    
    # 1. Convergence curves
    if viz_config.get('convergence_curve') and histories:
        try:
            plt.figure(figsize=(10, 6))
            fitness_histories = []
            for history in histories:
                if isinstance(history, dict) and 'fitness' in history:
                    fitness_histories.append(history['fitness'])
                elif isinstance(history, (list, np.ndarray)):
                    fitness_histories.append(history)
            
            if fitness_histories:
                for fitness_history in fitness_histories:
                    plt.semilogy(fitness_history, alpha=0.3, color='blue')
                
                histories_array = np.array(fitness_histories)
                mean_history = np.mean(histories_array, axis=0)
                plt.semilogy(mean_history, 'r-', linewidth=2, label='Mean')
                
                plt.xlabel('Iteration')
                plt.ylabel('Best Fitness (log scale)')
                plt.title(f'{algo.__name__} on {func_name} ({num_runs} runs)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                filename = f"{folder}/convergence_multiple_runs.png"
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                print(f"✓ Convergence (multiple runs): {filename}")
                plots_generated += 1
            plt.close()
        except Exception as e:
            print(f"✗ Convergence plot failed: {e}")
    
    # 2. Boxplot
    if viz_config.get('fitness_boxplot'):
        try:
            plt.figure(figsize=(8, 6))
            plt.boxplot(fitness, vert=True)
            plt.ylabel('Best Fitness')
            plt.title(f'Distribution of Final Results\n{algo.__name__} on {func_name}')
            plt.grid(True, alpha=0.3)
            filename = f"{folder}/boxplot_distribution.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✓ Boxplot distribution: {filename}")
            plots_generated += 1
            plt.close()
        except Exception as e:
            print(f"✗ Boxplot failed: {e}")
    
    # 3. Violin plot
    if viz_config.get('fitness_boxplot') and len(fitness) >= 5:
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.violinplot([fitness], vert=True, showmeans=True, showmedians=True)
            ax.set_ylabel('Best Fitness')
            ax.set_title(f'Violin Plot - {algo.__name__} on {func_name}')
            ax.grid(True, alpha=0.3, axis='y')
            filename = f"{folder}/violin_plot.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✓ Violin plot: {filename}")
            plots_generated += 1
            plt.close()
        except Exception as e:
            print(f"✗ Violin plot failed: {e}")
    
    # 4. Statistical summary
    if viz_config.get('fitness_boxplot'):
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            stats = {
                'Best': np.min(fitness),
                'Q1': np.percentile(fitness, 25),
                'Median': np.median(fitness),
                'Mean': np.mean(fitness),
                'Q3': np.percentile(fitness, 75),
                'Worst': np.max(fitness)
            }
            ax.barh(list(stats.keys()), list(stats.values()), alpha=0.7, edgecolor='black')
            ax.set_xlabel('Fitness Value')
            ax.set_title(f'Statistical Summary - {algo.__name__} on {func_name}')
            ax.grid(True, alpha=0.3, axis='x')
            filename = f"{folder}/statistical_summary.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✓ Statistical summary: {filename}")
            plots_generated += 1
            plt.close()
        except Exception as e:
            print(f"✗ Statistical summary failed: {e}")
    
    # 5. Convergence uncertainty
    if viz_config.get('convergence_curve') and histories:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            fitness_histories = []
            for history in histories:
                if isinstance(history, dict) and 'fitness' in history:
                    fitness_histories.append(history['fitness'])
                elif isinstance(history, (list, np.ndarray)):
                    fitness_histories.append(history)
            
            if fitness_histories:
                histories_array = np.array(fitness_histories)
                mean_history = np.mean(histories_array, axis=0)
                std_history = np.std(histories_array, axis=0)
                iterations = range(len(mean_history))
                
                ax.plot(iterations, mean_history, 'b-', linewidth=2, label='Mean')
                ax.fill_between(iterations, mean_history - std_history, 
                               mean_history + std_history,
                               alpha=0.3, color='blue', label='±1 Std Dev')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Best Fitness')
                ax.set_title(f'Convergence with Uncertainty - {func_name}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
                filename = f"{folder}/convergence_uncertainty.png"
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                print(f"✓ Convergence uncertainty: {filename}")
                plots_generated += 1
            plt.close()
        except Exception as e:
            print(f"✗ Convergence uncertainty failed: {e}")
    
    print(f"\nTotal plots generated: {plots_generated}")
    print("=" * 80)
    return plots_generated


def _generate_comparison_visualizations(folder, results, histories, func_name, viz_config):
    """Generate visualizations for algorithm comparison."""
    viz_config = _parse_viz_config(viz_config)
    
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    plots_generated = 0
    
    # 1. Convergence comparison
    if viz_config.get('convergence_curve') and histories:
        try:
            plt.figure(figsize=(12, 6))
            colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
            
            for (name, hist), color in zip(histories.items(), colors):
                mean_history = np.mean(hist, axis=0)
                std_history = np.std(hist, axis=0)
                iterations = np.arange(len(mean_history))
                
                plt.semilogy(iterations, mean_history, label=name, color=color, linewidth=2)
                plt.fill_between(iterations, mean_history - std_history, 
                               mean_history + std_history, alpha=0.2, color=color)
            
            plt.xlabel('Iteration')
            plt.ylabel('Best Fitness (log scale)')
            plt.title(f'Convergence Comparison - {func_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            filename = f"{folder}/convergence_comparison.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✓ Convergence comparison: {filename}")
            plots_generated += 1
            plt.close()
        except Exception as e:
            print(f"✗ Convergence comparison failed: {e}")
    
    # 2. Statistical comparison
    if viz_config.get('fitness_boxplot'):
        try:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.boxplot(results.values(), tick_labels=results.keys())
            plt.ylabel('Best Fitness')
            plt.title('Algorithm Comparison - Distribution')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
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
            filename = f"{folder}/statistical_comparison.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✓ Statistical comparison: {filename}")
            plots_generated += 1
            plt.close()
        except Exception as e:
            print(f"✗ Statistical comparison failed: {e}")
    
    print(f"\nTotal plots generated: {plots_generated}")
    print("=" * 80)
    return plots_generated
