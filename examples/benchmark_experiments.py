"""
Comprehensive benchmark experiments with automatic visualization.

This script runs experiments on multiple benchmark functions and generates:
- Convergence plots for each objective function
- Boxplots comparing performance across functions
- All results saved in timestamped experiment folders
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from pathlib import Path
from pyrade import DifferentialEvolution, OptimizationVisualizer
from pyrade.benchmarks import (
    Sphere, Rastrigin, Rosenbrock, Ackley, Griewank, 
    Schwefel, Levy, Michalewicz, Zakharov
)


class BenchmarkExperiment:
    """Run and visualize benchmark experiments."""
    
    def __init__(self, 
                 n_runs: int = 30,
                 population_size: int = 50,
                 max_iterations: int = 100,
                 dimensions: int = 10):
        """
        Initialize experiment parameters.
        
        Parameters
        ----------
        n_runs : int
            Number of independent runs per benchmark
        population_size : int
            DE population size
        max_iterations : int
            Maximum iterations per run
        dimensions : int
            Problem dimensionality
        """
        self.n_runs = n_runs
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.dimensions = dimensions
        
        # Create timestamped experiment folder
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_folder = Path(f"experiment_{timestamp}")
        self.experiment_folder.mkdir(exist_ok=True)
        
        print(f"Experiment folder created: {self.experiment_folder}")
        
        # Initialize visualizer
        self.viz = OptimizationVisualizer(figsize=(10, 6))
        
        # Define benchmarks
        self.benchmarks = {
            'Sphere': (Sphere(), [(-100, 100)] * self.dimensions),
            'Rastrigin': (Rastrigin(), [(-5.12, 5.12)] * self.dimensions),
            'Rosenbrock': (Rosenbrock(), [(-5, 10)] * self.dimensions),
            'Ackley': (Ackley(), [(-32.768, 32.768)] * self.dimensions),
            'Griewank': (Griewank(), [(-600, 600)] * self.dimensions),
            'Schwefel': (Schwefel(), [(-500, 500)] * self.dimensions),
            'Levy': (Levy(), [(-10, 10)] * self.dimensions),
            'Zakharov': (Zakharov(), [(-5, 10)] * self.dimensions),
        }
        
        # Storage for results
        self.results = {}
        
    def run_single_experiment(self, func_name, obj_func, bounds, run_id):
        """
        Run single optimization experiment.
        
        Parameters
        ----------
        func_name : str
            Name of benchmark function
        obj_func : callable
            Objective function
        bounds : list
            Variable bounds
        run_id : int
            Run identifier
            
        Returns
        -------
        dict
            Optimization results
        """
        de = DifferentialEvolution(
            obj_func=obj_func,
            bounds=np.array(bounds),
            population_size=self.population_size,
            max_iterations=self.max_iterations,
            F=0.8,
            CR=0.9,
            seed=run_id
        )
        
        result = de.run()
        return result
    
    def run_all_benchmarks(self):
        """Run experiments on all benchmark functions."""
        print("\n" + "=" * 70)
        print(f"RUNNING EXPERIMENTS: {self.n_runs} runs × {len(self.benchmarks)} functions")
        print("=" * 70 + "\n")
        
        for func_name, (obj_func, bounds) in self.benchmarks.items():
            print(f"Running {func_name}...")
            
            convergence_histories = []
            final_fitness_values = []
            best_solutions = []
            
            for run in range(self.n_runs):
                result = self.run_single_experiment(func_name, obj_func, bounds, run)
                
                # Extract results
                final_fitness_values.append(result['best_fitness'])
                best_solutions.append(result['best_solution'])
                
                # Get convergence history if available
                if 'fitness_history' in result:
                    convergence_histories.append(result['fitness_history'])
                else:
                    # Create simple history if not available
                    convergence_histories.append([result['best_fitness']] * self.max_iterations)
                
                if (run + 1) % 10 == 0:
                    print(f"  Completed {run + 1}/{self.n_runs} runs")
            
            # Store results
            self.results[func_name] = {
                'convergence_histories': convergence_histories,
                'final_fitness': final_fitness_values,
                'best_solutions': best_solutions,
                'mean_fitness': np.mean(final_fitness_values),
                'std_fitness': np.std(final_fitness_values),
                'min_fitness': np.min(final_fitness_values),
                'max_fitness': np.max(final_fitness_values),
                'median_fitness': np.median(final_fitness_values)
            }
            
            print(f"  ✓ {func_name} completed: Mean={self.results[func_name]['mean_fitness']:.6e}, "
                  f"Std={self.results[func_name]['std_fitness']:.6e}\n")
    
    def plot_convergence_for_each_function(self):
        """Generate convergence plot for each benchmark function."""
        print("Generating convergence plots for each function...")
        
        # Create subfolder for convergence plots
        conv_folder = self.experiment_folder / "convergence_plots"
        conv_folder.mkdir(exist_ok=True)
        
        for func_name, data in self.results.items():
            histories = data['convergence_histories']
            
            # Plot with standard deviation
            fig = self.viz.plot_convergence_curve(
                history=histories,
                title=f"Convergence Curve: {func_name} Function (D={self.dimensions})",
                xlabel="Generation",
                ylabel="Best Fitness (log scale)",
                log_scale=True,
                show_std=True,
                save_path=conv_folder / f"{func_name.lower()}_convergence.png"
            )
            plt.close(fig)
            
        print(f"  ✓ Saved {len(self.results)} convergence plots to {conv_folder}\n")
    
    def plot_all_convergences_combined(self):
        """Plot all convergence curves on one figure for comparison."""
        print("Generating combined convergence plot...")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for func_name, data in self.results.items():
            histories = np.array(data['convergence_histories'])
            mean_history = np.mean(histories, axis=0)
            generations = range(len(mean_history))
            
            ax.plot(generations, mean_history, marker='o', markersize=2,
                   linewidth=2, label=func_name, alpha=0.8)
        
        ax.set_xlabel("Generation", fontsize=12)
        ax.set_ylabel("Mean Best Fitness (log scale)", fontsize=12)
        ax.set_title(f"Convergence Comparison - All Functions (D={self.dimensions}, "
                    f"{self.n_runs} runs)", fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')
        
        plt.tight_layout()
        save_path = self.experiment_folder / "all_functions_convergence.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✓ Saved combined convergence plot to {save_path}\n")
    
    def plot_fitness_boxplots(self):
        """Generate boxplots comparing final fitness distributions."""
        print("Generating fitness distribution boxplots...")
        
        # Prepare data for boxplot
        fitness_data = {
            func_name: data['final_fitness']
            for func_name, data in self.results.items()
        }
        
        # Create boxplot
        fig = self.viz.plot_fitness_boxplot(
            fitness_data=fitness_data,
            title=f"Final Fitness Distribution Across Benchmarks\n"
                  f"(D={self.dimensions}, Pop={self.population_size}, "
                  f"Iter={self.max_iterations}, Runs={self.n_runs})",
            ylabel="Final Best Fitness",
            save_path=self.experiment_folder / "fitness_boxplot.png"
        )
        plt.close(fig)
        
        print(f"  ✓ Saved boxplot to {self.experiment_folder / 'fitness_boxplot.png'}\n")
    
    def plot_fitness_boxplots_separate_scales(self):
        """Generate separate boxplots for functions with different scales."""
        print("Generating boxplots with separate scales...")
        
        # Group functions by fitness magnitude
        fitness_magnitudes = {
            func_name: np.log10(np.mean(data['final_fitness']) + 1e-10)
            for func_name, data in self.results.items()
        }
        
        # Sort and split into groups
        sorted_funcs = sorted(fitness_magnitudes.items(), key=lambda x: x[1])
        n_funcs = len(sorted_funcs)
        mid_point = n_funcs // 2
        
        # Group 1: Lower magnitude functions
        group1_data = {name: self.results[name]['final_fitness'] 
                      for name, _ in sorted_funcs[:mid_point]}
        
        # Group 2: Higher magnitude functions
        group2_data = {name: self.results[name]['final_fitness'] 
                      for name, _ in sorted_funcs[mid_point:]}
        
        # Plot group 1
        if group1_data:
            fig1 = self.viz.plot_fitness_boxplot(
                fitness_data=group1_data,
                title=f"Final Fitness Distribution - Group 1 (Lower Scale)\n"
                      f"(Runs={self.n_runs})",
                ylabel="Final Best Fitness",
                save_path=self.experiment_folder / "fitness_boxplot_group1.png"
            )
            plt.close(fig1)
        
        # Plot group 2
        if group2_data:
            fig2 = self.viz.plot_fitness_boxplot(
                fitness_data=group2_data,
                title=f"Final Fitness Distribution - Group 2 (Higher Scale)\n"
                      f"(Runs={self.n_runs})",
                ylabel="Final Best Fitness",
                save_path=self.experiment_folder / "fitness_boxplot_group2.png"
            )
            plt.close(fig2)
        
        print(f"  ✓ Saved grouped boxplots\n")
    
    def generate_statistics_table(self):
        """Generate and save statistics table."""
        print("Generating statistics table...")
        
        # Create statistics file
        stats_file = self.experiment_folder / "statistics.txt"
        
        with open(stats_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"BENCHMARK EXPERIMENT STATISTICS\n")
            f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Configuration:\n")
            f.write(f"  - Dimensions: {self.dimensions}\n")
            f.write(f"  - Population Size: {self.population_size}\n")
            f.write(f"  - Max Iterations: {self.max_iterations}\n")
            f.write(f"  - Number of Runs: {self.n_runs}\n")
            f.write(f"  - Total Evaluations per Run: "
                   f"{self.population_size * self.max_iterations}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write(f"{'Function':<15} {'Mean':<15} {'Std':<15} {'Min':<15} {'Median':<15} {'Max':<15}\n")
            f.write("-" * 80 + "\n")
            
            for func_name, data in sorted(self.results.items()):
                f.write(f"{func_name:<15} "
                       f"{data['mean_fitness']:<15.6e} "
                       f"{data['std_fitness']:<15.6e} "
                       f"{data['min_fitness']:<15.6e} "
                       f"{data['median_fitness']:<15.6e} "
                       f"{data['max_fitness']:<15.6e}\n")
            
            f.write("-" * 80 + "\n\n")
            
            # Add ranking by mean fitness
            f.write("Ranking by Mean Fitness:\n")
            ranked = sorted(self.results.items(), key=lambda x: x[1]['mean_fitness'])
            for rank, (func_name, data) in enumerate(ranked, 1):
                f.write(f"  {rank}. {func_name}: {data['mean_fitness']:.6e}\n")
        
        print(f"  ✓ Saved statistics to {stats_file}\n")
        
        # Also print to console
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        with open(stats_file, 'r') as f:
            print(f.read())
    
    def save_raw_data(self):
        """Save raw experimental data."""
        print("Saving raw data...")
        
        # Create data subfolder
        data_folder = self.experiment_folder / "raw_data"
        data_folder.mkdir(exist_ok=True)
        
        for func_name, data in self.results.items():
            # Save convergence histories
            conv_file = data_folder / f"{func_name.lower()}_convergence.npy"
            np.save(conv_file, np.array(data['convergence_histories']))
            
            # Save final fitness values
            fitness_file = data_folder / f"{func_name.lower()}_final_fitness.npy"
            np.save(fitness_file, np.array(data['final_fitness']))
            
            # Save best solutions
            solutions_file = data_folder / f"{func_name.lower()}_best_solutions.npy"
            np.save(solutions_file, np.array(data['best_solutions']))
        
        print(f"  ✓ Saved raw data to {data_folder}\n")
    
    def run_complete_experiment(self):
        """Run complete experimental pipeline."""
        start_time = datetime.now()
        
        print("\n" + "=" * 80)
        print("BENCHMARK EXPERIMENT SUITE")
        print("=" * 80)
        print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Experiment Folder: {self.experiment_folder}")
        print("=" * 80 + "\n")
        
        # Run experiments
        self.run_all_benchmarks()
        
        # Generate visualizations
        self.plot_convergence_for_each_function()
        self.plot_all_convergences_combined()
        self.plot_fitness_boxplots()
        self.plot_fitness_boxplots_separate_scales()
        
        # Generate statistics
        self.generate_statistics_table()
        
        # Save raw data
        self.save_raw_data()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETED!")
        print("=" * 80)
        print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Duration: {duration}")
        print(f"\nAll results saved to: {self.experiment_folder.absolute()}")
        print("\nGenerated files:")
        print(f"  - convergence_plots/ ({len(self.benchmarks)} individual plots)")
        print(f"  - all_functions_convergence.png")
        print(f"  - fitness_boxplot.png")
        print(f"  - fitness_boxplot_group1.png")
        print(f"  - fitness_boxplot_group2.png")
        print(f"  - statistics.txt")
        print(f"  - raw_data/ (convergence, fitness, solutions for each function)")
        print("=" * 80 + "\n")


def main():
    """Main execution function."""
    # Create and run experiment
    experiment = BenchmarkExperiment(
        n_runs=30,              # 30 independent runs
        population_size=50,     # Population size
        max_iterations=100,     # Maximum iterations
        dimensions=10           # Problem dimensionality
    )
    
    # Run complete experimental pipeline
    experiment.run_complete_experiment()


def quick_test():
    """Quick test with fewer runs for debugging."""
    print("Running quick test (5 runs, 3 functions)...")
    
    experiment = BenchmarkExperiment(
        n_runs=5,
        population_size=30,
        max_iterations=50,
        dimensions=5
    )
    
    # Test with only 3 functions
    experiment.benchmarks = {
        'Sphere': experiment.benchmarks['Sphere'],
        'Rastrigin': experiment.benchmarks['Rastrigin'],
        'Rosenbrock': experiment.benchmarks['Rosenbrock'],
    }
    
    experiment.run_complete_experiment()


if __name__ == "__main__":
    # Run full experiment
    main()
    
    # Or run quick test:
    # quick_test()
