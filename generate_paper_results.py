"""
Paper Results Generator - Original DE on Standard Benchmarks
Runs classic DErand1bin algorithm on benchmark functions and generates all visualizations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pyrade.benchmarks.functions import (
    Sphere, Rastrigin, Rosenbrock, Ackley, Griewank, 
    Schwefel, Zakharov
)
from pyrade.core.algorithm import DifferentialEvolution
from pyrade.operators.mutation import DErand1
from pyrade.operators.crossover import BinomialCrossover
from pyrade.operators.selection import ElitistSelection
from pyrade.visualization import OptimizationVisualizer

# Create results directory
RESULTS_DIR = Path("paper_result")
RESULTS_DIR.mkdir(exist_ok=True)

# Benchmark functions to test
BENCHMARKS = [
    ("Sphere", Sphere(), 30, (-100, 100)),
    ("Rastrigin", Rastrigin(), 30, (-5.12, 5.12)),
    ("Rosenbrock", Rosenbrock(), 30, (-5, 10)),
    ("Ackley", Ackley(), 30, (-32.768, 32.768)),
    ("Griewank", Griewank(), 30, (-600, 600)),
    ("Schwefel", Schwefel(), 30, (-500, 500)),
    ("Zakharov", Zakharov(), 30, (-5, 10)),
]

# DE parameters (classic settings)
CONFIG = {
    'pop_size': 50,
    'F': 0.8,
    'CR': 0.9,
    'max_iter': 500,
    'n_runs': 30,
}

def run_single_benchmark(name, func, dim, bounds):
    """Run DE on a single benchmark function and generate all visualizations."""
    
    print(f"\n{'='*60}")
    print(f"Running: {name} (Dimension: {dim})")
    print(f"{'='*60}")
    
    # Create subfolder for this benchmark
    benchmark_dir = RESULTS_DIR / name.lower()
    benchmark_dir.mkdir(exist_ok=True)
    
    # Store results from multiple runs
    all_best_fitness = []
    all_convergence = []
    final_populations = []
    final_fitness_values = []
    
    # Run multiple times
    for run in range(CONFIG['n_runs']):
        print(f"  Run {run+1}/{CONFIG['n_runs']}...", end=' ')
        
        # Initialize DE with classic components
        de = DifferentialEvolution(
            objective_func=func,
            bounds=np.array([bounds] * dim),
            pop_size=CONFIG['pop_size'],
            mutation=DErand1(F=CONFIG['F']),
            crossover=BinomialCrossover(CR=CONFIG['CR']),
            selection=ElitistSelection(),
            max_iter=CONFIG['max_iter'],
            seed=run
        )
        
        # Run optimization
        result = de.optimize()
        
        all_best_fitness.append(result['best_fitness'])
        all_convergence.append(result['history']['fitness'])
        final_populations.append(result['best_solution'])
        final_fitness_values.append(result['best_fitness'])
        
        print(f"Best Fitness: {result['best_fitness']:.6e}")
    
    # Convert to arrays
    all_convergence = np.array(all_convergence)
    all_best_fitness = np.array(all_best_fitness)
    
    # Generate visualizations
    print(f"\n  Generating visualizations for {name}...")
    
    # 1. Convergence curves (with statistics)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot individual runs (lighter)
    for conv in all_convergence:
        ax.semilogy(conv, alpha=0.1, color='blue')
    
    # Plot mean convergence
    mean_conv = np.mean(all_convergence, axis=0)
    std_conv = np.std(all_convergence, axis=0)
    generations = np.arange(len(mean_conv))
    
    ax.semilogy(mean_conv, 'r-', linewidth=2, label='Mean')
    ax.fill_between(generations, 
                     mean_conv - std_conv, 
                     mean_conv + std_conv, 
                     alpha=0.3, color='red', label='±1 Std Dev')
    
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Best Fitness (log scale)', fontsize=12)
    ax.set_title(f'Convergence Curve - {name} Function', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(benchmark_dir / 'convergence_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Box plot of final fitness values
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot([all_best_fitness], labels=[name], widths=0.5)
    ax.set_ylabel('Final Best Fitness', fontsize=12)
    ax.set_title(f'Fitness Distribution - {name} Function', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = f'Mean: {np.mean(all_best_fitness):.6e}\n'
    stats_text += f'Median: {np.median(all_best_fitness):.6e}\n'
    stats_text += f'Std: {np.std(all_best_fitness):.6e}\n'
    stats_text += f'Min: {np.min(all_best_fitness):.6e}\n'
    stats_text += f'Max: {np.max(all_best_fitness):.6e}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(benchmark_dir / 'fitness_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Histogram of final fitness
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_best_fitness, bins=15, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(all_best_fitness), color='red', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(np.median(all_best_fitness), color='green', linestyle='--', linewidth=2, label='Median')
    ax.set_xlabel('Final Best Fitness', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Histogram of Final Fitness - {name} Function', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(benchmark_dir / 'fitness_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Landscape plot (2D projection) - only for first two dimensions
    if dim >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create mesh
        x = np.linspace(bounds[0], bounds[1], 100)
        y = np.linspace(bounds[0], bounds[1], 100)
        X, Y = np.meshgrid(x, y)
        
        # Evaluate function (fix other dimensions at 0)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = np.zeros(dim)
                point[0] = X[i, j]
                point[1] = Y[i, j]
                Z[i, j] = func(point)
        
        # Plot contour
        contour = ax.contourf(X, Y, Z, levels=30, cmap='viridis')
        plt.colorbar(contour, ax=ax, label='Fitness Value')
        
        # Overlay best solutions from all runs
        best_x = [pop[0] for pop in final_populations]
        best_y = [pop[1] for pop in final_populations]
        ax.scatter(best_x, best_y, c='red', marker='*', s=200, 
                  edgecolors='white', linewidths=1.5, label='Best Solutions', zorder=5)
        
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_title(f'Landscape Plot (2D Projection) - {name} Function', fontsize=14, fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(benchmark_dir / 'landscape_2d.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Statistical summary plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: Convergence curve
    ax = axes[0, 0]
    ax.semilogy(mean_conv, 'b-', linewidth=2)
    ax.fill_between(generations, mean_conv - std_conv, mean_conv + std_conv, alpha=0.3)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness (log)')
    ax.set_title('Mean Convergence')
    ax.grid(True, alpha=0.3)
    
    # Subplot 2: Fitness distribution
    ax = axes[0, 1]
    ax.boxplot([all_best_fitness], widths=0.5)
    ax.set_ylabel('Final Fitness')
    ax.set_title('Final Fitness Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Subplot 3: Success rate over generations (reaching thresholds)
    ax = axes[1, 0]
    thresholds = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    for threshold in thresholds:
        success_rate = np.mean(all_convergence <= threshold, axis=0) * 100
        ax.plot(success_rate, label=f'≤ {threshold:.0e}')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate at Different Thresholds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Subplot 4: Statistics table
    ax = axes[1, 1]
    ax.axis('off')
    stats_data = [
        ['Metric', 'Value'],
        ['Mean', f'{np.mean(all_best_fitness):.6e}'],
        ['Median', f'{np.median(all_best_fitness):.6e}'],
        ['Std Dev', f'{np.std(all_best_fitness):.6e}'],
        ['Min', f'{np.min(all_best_fitness):.6e}'],
        ['Max', f'{np.max(all_best_fitness):.6e}'],
        ['Runs', f'{CONFIG["n_runs"]}'],
        ['Generations', f'{CONFIG["max_iter"]}'],
        ['Pop Size', f'{CONFIG["pop_size"]}'],
        ['F', f'{CONFIG["F"]}'],
        ['CR', f'{CONFIG["CR"]}'],
    ]
    table = ax.table(cellText=stats_data, cellLoc='left', loc='center',
                     colWidths=[0.4, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle(f'{name} Function - Statistical Summary', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(benchmark_dir / 'statistical_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical results to CSV
    results_csv = benchmark_dir / 'numerical_results.csv'
    with open(results_csv, 'w') as f:
        f.write('Run,Best_Fitness\n')
        for i, fitness in enumerate(all_best_fitness, 1):
            f.write(f'{i},{fitness:.15e}\n')
        f.write(f'\nStatistics\n')
        f.write(f'Mean,{np.mean(all_best_fitness):.15e}\n')
        f.write(f'Median,{np.median(all_best_fitness):.15e}\n')
        f.write(f'Std,{np.std(all_best_fitness):.15e}\n')
        f.write(f'Min,{np.min(all_best_fitness):.15e}\n')
        f.write(f'Max,{np.max(all_best_fitness):.15e}\n')
    
    print(f"  ✓ Results saved to: {benchmark_dir}")
    print(f"  ✓ Generated 5 visualizations + numerical data")

def generate_comparison_plot():
    """Generate a comparison plot across all benchmarks."""
    print(f"\n{'='*60}")
    print("Generating Overall Comparison Plot")
    print(f"{'='*60}")
    
    benchmark_names = []
    mean_fitness = []
    std_fitness = []
    
    # Collect results from each benchmark
    for name, _, _, _ in BENCHMARKS:
        results_csv = RESULTS_DIR / name.lower() / 'numerical_results.csv'
        if results_csv.exists():
            # Read numerical results
            fitness_values = []
            with open(results_csv, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:  # Skip header
                    if line.strip() and not line.startswith('Statistics') and ',' in line:
                        parts = line.strip().split(',')
                        if len(parts) == 2 and parts[0].isdigit():
                            fitness_values.append(float(parts[1]))
            
            if fitness_values:
                benchmark_names.append(name)
                mean_fitness.append(np.mean(fitness_values))
                std_fitness.append(np.std(fitness_values))
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Bar chart with error bars
    x_pos = np.arange(len(benchmark_names))
    bars = ax1.bar(x_pos, mean_fitness, yerr=std_fitness, capsize=5, 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Color bars by performance
    colors = plt.cm.viridis(np.linspace(0, 1, len(benchmark_names)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(benchmark_names, rotation=45, ha='right')
    ax1.set_ylabel('Mean Best Fitness (log scale)', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_title('Benchmark Comparison - Mean Fitness', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Box plots
    all_fitness_data = []
    for name, _, _, _ in BENCHMARKS:
        results_csv = RESULTS_DIR / name.lower() / 'numerical_results.csv'
        if results_csv.exists():
            fitness_values = []
            with open(results_csv, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    if line.strip() and not line.startswith('Statistics') and ',' in line:
                        parts = line.strip().split(',')
                        if len(parts) == 2 and parts[0].isdigit():
                            fitness_values.append(float(parts[1]))
            all_fitness_data.append(fitness_values)
    
    bp = ax2.boxplot(all_fitness_data, labels=benchmark_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_xticklabels(benchmark_names, rotation=45, ha='right')
    ax2.set_ylabel('Final Best Fitness (log scale)', fontsize=12)
    ax2.set_yscale('log')
    ax2.set_title('Benchmark Comparison - Fitness Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Original DE (DErand1bin) - Benchmark Comparison', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'overall_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Comparison plot saved to: {RESULTS_DIR / 'overall_comparison.png'}")

def generate_combined_figures(all_results):
    """Generate combined figures with all benchmarks as subplots."""
    print(f"\n{'='*60}")
    print("Generating Combined Subfigures")
    print(f"{'='*60}")
    
    n_benchmarks = len(all_results)
    
    # 1. Combined Convergence Curves
    print("  Creating combined convergence curves...")
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for idx, (name, data) in enumerate(all_results.items()):
        ax = axes[idx]
        all_conv = data['convergence']
        
        # Plot individual runs (lighter)
        for conv in all_conv:
            ax.semilogy(conv, alpha=0.1, color='blue', linewidth=0.5)
        
        # Plot mean convergence
        mean_conv = np.mean(all_conv, axis=0)
        std_conv = np.std(all_conv, axis=0)
        generations = np.arange(len(mean_conv))
        
        ax.semilogy(mean_conv, 'r-', linewidth=2, label='Mean')
        ax.fill_between(generations, 
                         np.maximum(mean_conv - std_conv, 1e-20), 
                         mean_conv + std_conv, 
                         alpha=0.3, color='red')
        
        ax.set_xlabel('Generation', fontsize=10)
        ax.set_ylabel('Best Fitness', fontsize=10)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # Hide extra subplot if odd number
    if n_benchmarks < len(axes):
        for idx in range(n_benchmarks, len(axes)):
            axes[idx].axis('off')
    
    plt.suptitle('Convergence Curves - All Benchmarks', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'combined_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: combined_convergence.png")
    
    # 2. Combined Box Plots
    print("  Creating combined fitness distributions...")
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for idx, (name, data) in enumerate(all_results.items()):
        ax = axes[idx]
        fitness_vals = data['final_fitness']
        
        bp = ax.boxplot([fitness_vals], widths=0.5, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        
        # Add statistics
        stats_text = f'Mean: {np.mean(fitness_vals):.3e}\n'
        stats_text += f'Std: {np.std(fitness_vals):.3e}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_ylabel('Final Fitness', fontsize=10)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels([''])
    
    if n_benchmarks < len(axes):
        for idx in range(n_benchmarks, len(axes)):
            axes[idx].axis('off')
    
    plt.suptitle('Fitness Distributions - All Benchmarks', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'combined_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: combined_distributions.png")
    
    # 3. Combined Histograms
    print("  Creating combined histograms...")
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for idx, (name, data) in enumerate(all_results.items()):
        ax = axes[idx]
        fitness_vals = data['final_fitness']
        
        ax.hist(fitness_vals, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(np.mean(fitness_vals), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.axvline(np.median(fitness_vals), color='green', linestyle='--', linewidth=2, label='Median')
        
        ax.set_xlabel('Final Fitness', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    if n_benchmarks < len(axes):
        for idx in range(n_benchmarks, len(axes)):
            axes[idx].axis('off')
    
    plt.suptitle('Fitness Histograms - All Benchmarks', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'combined_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: combined_histograms.png")
    
    # 4. Combined Landscapes
    print("  Creating combined landscape plots...")
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for idx, (name, data) in enumerate(all_results.items()):
        ax = axes[idx]
        func = data['function']
        bounds = data['bounds']
        dim = data['dim']
        best_solutions = data['best_solutions']
        
        # Create mesh
        x = np.linspace(bounds[0], bounds[1], 80)
        y = np.linspace(bounds[0], bounds[1], 80)
        X, Y = np.meshgrid(x, y)
        
        # Evaluate function
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = np.zeros(dim)
                point[0] = X[i, j]
                point[1] = Y[i, j]
                Z[i, j] = func(point)
        
        # Plot contour
        contour = ax.contourf(X, Y, Z, levels=25, cmap='viridis')
        
        # Overlay best solutions
        best_x = [sol[0] for sol in best_solutions]
        best_y = [sol[1] for sol in best_solutions]
        ax.scatter(best_x, best_y, c='red', marker='*', s=100, 
                  edgecolors='white', linewidths=1, zorder=5)
        
        ax.set_xlabel('$x_1$', fontsize=10)
        ax.set_ylabel('$x_2$', fontsize=10)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
    
    if n_benchmarks < len(axes):
        for idx in range(n_benchmarks, len(axes)):
            axes[idx].axis('off')
    
    plt.suptitle('Landscape Plots (2D Projection) - All Benchmarks', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'combined_landscapes.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: combined_landscapes.png")
    
    print(f"  ✓ All combined figures generated!")

def main():
    """Main execution function."""
    print("\n" + "="*60)
    print(" PAPER RESULTS GENERATION - ORIGINAL DE")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Algorithm: Original DE (DErand1/bin)")
    print(f"  Population Size: {CONFIG['pop_size']}")
    print(f"  F (Mutation Factor): {CONFIG['F']}")
    print(f"  CR (Crossover Rate): {CONFIG['CR']}")
    print(f"  Max Generations: {CONFIG['max_iter']}")
    print(f"  Independent Runs: {CONFIG['n_runs']}")
    print(f"  Output Directory: {RESULTS_DIR.absolute()}")
    print(f"\nBenchmarks: {len(BENCHMARKS)} functions")
    for name, _, dim, bounds in BENCHMARKS:
        print(f"  - {name} (dim={dim}, bounds={bounds})")
    
    # Store results for combined figures
    all_results = {}
    
    # Run each benchmark
    for name, func, dim, bounds in BENCHMARKS:
        run_single_benchmark(name, func, dim, bounds)
        
        # Load results for combined figures
        results_csv = RESULTS_DIR / name.lower() / 'numerical_results.csv'
        fitness_values = []
        with open(results_csv, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                if line.strip() and not line.startswith('Statistics') and ',' in line:
                    parts = line.strip().split(',')
                    if len(parts) == 2 and parts[0].isdigit():
                        fitness_values.append(float(parts[1]))
        
        # Re-run to get convergence data (simplified)
        print(f"  Collecting convergence data for combined figures...")
        all_convergence = []
        best_solutions = []
        for run in range(min(5, CONFIG['n_runs'])):  # Use first 5 runs for landscapes
            de = DifferentialEvolution(
                objective_func=func,
                bounds=np.array([bounds] * dim),
                pop_size=CONFIG['pop_size'],
                mutation=DErand1(F=CONFIG['F']),
                crossover=BinomialCrossover(CR=CONFIG['CR']),
                selection=ElitistSelection(),
                max_iter=CONFIG['max_iter'],
                seed=run
            )
            result = de.optimize()
            all_convergence.append(result['history']['fitness'])
            best_solutions.append(result['best_solution'])
        
        all_results[name] = {
            'convergence': all_convergence,
            'final_fitness': fitness_values,
            'best_solutions': best_solutions,
            'function': func,
            'bounds': bounds,
            'dim': dim
        }
    
    # Generate comparison plot
    generate_comparison_plot()
    
    # Generate combined subfigures
    generate_combined_figures(all_results)
    
    print(f"\n{'='*60}")
    print(" COMPLETE!")
    print(f"{'='*60}")
    print(f"\nAll results saved to: {RESULTS_DIR.absolute()}")
    print(f"\nGenerated files per benchmark:")
    print(f"  1. convergence_curve.png - Convergence with statistics")
    print(f"  2. fitness_distribution.png - Box plot with stats")
    print(f"  3. fitness_histogram.png - Distribution histogram")
    print(f"  4. landscape_2d.png - 2D landscape visualization")
    print(f"  5. statistical_summary.png - Comprehensive summary")
    print(f"  6. numerical_results.csv - Raw numerical data")
    print(f"\nCombined figures (all benchmarks):")
    print(f"  1. combined_convergence.png - All convergence curves")
    print(f"  2. combined_distributions.png - All box plots")
    print(f"  3. combined_histograms.png - All histograms")
    print(f"  4. combined_landscapes.png - All landscape plots")
    print(f"\nOverall comparison:")
    print(f"  - overall_comparison.png - Cross-benchmark comparison")
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
