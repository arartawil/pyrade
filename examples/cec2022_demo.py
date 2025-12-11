"""
CEC 2022 Benchmark Usage Example

This example demonstrates how to use the CEC 2022 test functions.

NOTE: You need to obtain the official CEC 2022 data files first.
See pyrade/benchmarks/CEC2022_README.md for details.
"""

import numpy as np
from pyrade.algorithms.adaptive import LSHADE, jSO
from pyrade.benchmarks import CEC2022, get_cec2022_bounds


def example_single_function():
    """Example: Optimize a single CEC 2022 function"""
    print("=" * 80)
    print("CEC 2022 Single Function Example")
    print("=" * 80)
    
    # Setup
    func_num = 1  # Zakharov function
    dim = 10
    
    # NOTE: Update this path to where you stored the CEC 2022 data files
    data_dir = 'path/to/cec2022_data'
    
    try:
        # Create benchmark function
        benchmark = CEC2022(func_num=func_num, dim=dim, data_dir=data_dir)
        bounds = get_cec2022_bounds(dim)
        
        print(f"\nFunction: F{func_num} (Zakharov)")
        print(f"Dimension: {dim}")
        print(f"Bounds: [{bounds[0, 0]}, {bounds[0, 1]}]")
        print(f"Optimum bias: {benchmark.bias[func_num-1]}")
        
        # Create algorithm
        algorithm = LSHADE(pop_size_init=100)
        
        # Run optimization
        print("\nRunning L-SHADE...")
        best_solution, best_fitness = algorithm.optimize(
            objective_func=benchmark,
            bounds=bounds,
            max_evals=10000,
            show_progress=True
        )
        
        print(f"\nResults:")
        print(f"Best fitness: {best_fitness:.6e}")
        print(f"Best solution (first 5): {best_solution[:5]}")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease download CEC 2022 data files first.")
        print("See pyrade/benchmarks/CEC2022_README.md for instructions.")


def example_multiple_functions():
    """Example: Test multiple CEC 2022 functions"""
    print("\n" + "=" * 80)
    print("CEC 2022 Multiple Functions Example")
    print("=" * 80)
    
    # Test functions 1-5 (basic functions)
    functions = [1, 2, 3, 4, 5]
    dim = 10
    data_dir = 'path/to/cec2022_data'
    
    try:
        algorithm = jSO(pop_size_init=50)
        
        results = {}
        for func_num in functions:
            print(f"\n{'='*60}")
            print(f"Testing F{func_num}")
            print('='*60)
            
            benchmark = CEC2022(func_num=func_num, dim=dim, data_dir=data_dir)
            bounds = get_cec2022_bounds(dim)
            
            best_solution, best_fitness = algorithm.optimize(
                objective_func=benchmark,
                bounds=bounds,
                max_evals=5000,
                show_progress=False
            )
            
            results[func_num] = best_fitness
            print(f"F{func_num} fitness: {best_fitness:.6e}")
        
        print("\n" + "=" * 80)
        print("Summary of Results:")
        print("=" * 80)
        for func_num, fitness in results.items():
            print(f"F{func_num}: {fitness:.6e}")
            
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease download CEC 2022 data files first.")


def example_algorithm_comparison():
    """Example: Compare algorithms on CEC 2022 functions"""
    print("\n" + "=" * 80)
    print("CEC 2022 Algorithm Comparison")
    print("=" * 80)
    
    func_num = 1
    dim = 10
    data_dir = 'path/to/cec2022_data'
    num_runs = 5
    
    try:
        benchmark = CEC2022(func_num=func_num, dim=dim, data_dir=data_dir)
        bounds = get_cec2022_bounds(dim)
        
        # Test different algorithms
        algorithms = {
            'LSHADE': LSHADE(pop_size_init=50),
            'jSO': jSO(pop_size_init=50),
        }
        
        results = {}
        
        for alg_name, algorithm in algorithms.items():
            print(f"\nTesting {alg_name}...")
            run_results = []
            
            for run in range(num_runs):
                best_solution, best_fitness = algorithm.optimize(
                    objective_func=benchmark,
                    bounds=bounds,
                    max_evals=5000,
                    show_progress=False
                )
                run_results.append(best_fitness)
                print(f"  Run {run+1}: {best_fitness:.6e}")
            
            results[alg_name] = {
                'mean': np.mean(run_results),
                'std': np.std(run_results),
                'best': np.min(run_results),
                'worst': np.max(run_results)
            }
        
        print("\n" + "=" * 80)
        print("Comparison Summary:")
        print("=" * 80)
        print(f"{'Algorithm':<15} {'Mean':<15} {'Std':<15} {'Best':<15} {'Worst':<15}")
        print("-" * 80)
        
        for alg_name, stats in results.items():
            print(f"{alg_name:<15} {stats['mean']:<15.6e} {stats['std']:<15.6e} "
                  f"{stats['best']:<15.6e} {stats['worst']:<15.6e}")
    
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease download CEC 2022 data files first.")


def example_without_data():
    """Example: Show how CEC 2022 works (without actual data files)"""
    print("\n" + "=" * 80)
    print("CEC 2022 Function Information (No Data Required)")
    print("=" * 80)
    
    function_info = {
        1: ("Zakharov", "Unimodal", 300),
        2: ("Rosenbrock", "Unimodal", 400),
        3: ("Schaffer F7", "Multimodal", 600),
        4: ("Step Rastrigin", "Multimodal", 800),
        5: ("Levy", "Multimodal", 900),
        6: ("Hybrid 1 (3 funcs)", "Hybrid", 1800),
        7: ("Hybrid 2 (6 funcs)", "Hybrid", 2000),
        8: ("Hybrid 3 (5 funcs)", "Hybrid", 2200),
        9: ("Composition 1 (5 funcs)", "Composition", 2300),
        10: ("Composition 2 (3 funcs)", "Composition", 2400),
        11: ("Composition 3 (5 funcs)", "Composition", 2600),
        12: ("Composition 4 (6 funcs)", "Composition", 2700),
    }
    
    print(f"\n{'Func':<6} {'Name':<25} {'Type':<15} {'Bias':<10} {'Dims'}")
    print("-" * 80)
    
    for func_num, (name, ftype, bias) in function_info.items():
        dims = "2, 10, 20" if func_num <= 5 or func_num >= 9 else "10, 20"
        print(f"F{func_num:<5} {name:<25} {ftype:<15} {bias:<10} {dims}")
    
    print("\n" + "=" * 80)
    print("To use CEC 2022 functions:")
    print("1. Download data files from official CEC 2022 competition website")
    print("2. Place files in a directory (e.g., 'cec2022_data/')")
    print("3. Initialize: CEC2022(func_num=1, dim=10, data_dir='cec2022_data')")
    print("4. See pyrade/benchmarks/CEC2022_README.md for full details")
    print("=" * 80)


if __name__ == "__main__":
    print("\nPyRADE CEC 2022 Examples")
    print("=" * 80)
    
    # Show function information (doesn't require data files)
    example_without_data()
    
    # Uncomment these after obtaining data files:
    # example_single_function()
    # example_multiple_functions()
    # example_algorithm_comparison()
    
    print("\n" + "=" * 80)
    print("Note: Actual optimization examples are commented out.")
    print("Uncomment them after obtaining CEC 2022 data files.")
    print("=" * 80)
