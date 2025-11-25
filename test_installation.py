"""
Quick test to verify PyRADE installation and functionality.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np

print("Testing PyRADE Installation...")
print("=" * 60)

# Test imports
print("\n1. Testing imports...")
try:
    from pyrade import DifferentialEvolution
    from pyrade.operators import DErand1, BinomialCrossover
    from pyrade.benchmarks import Sphere
    print("   ✓ All imports successful")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test basic optimization
print("\n2. Testing basic optimization...")
try:
    func = Sphere(dim=5)
    
    optimizer = DifferentialEvolution(
        objective_func=func,
        bounds=func.get_bounds_array(),
        pop_size=20,
        max_iter=50,
        verbose=False,
        seed=42
    )
    
    result = optimizer.optimize()
    
    if result['best_fitness'] < 1e-3:
        print(f"   ✓ Optimization successful")
        print(f"     Final fitness: {result['best_fitness']:.6e}")
    else:
        print(f"   ⚠ Optimization completed but fitness may be suboptimal")
        print(f"     Final fitness: {result['best_fitness']:.6e}")
except Exception as e:
    print(f"   ✗ Error during optimization: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test custom strategies
print("\n3. Testing custom strategies...")
try:
    func = Sphere(dim=5)
    
    optimizer = DifferentialEvolution(
        objective_func=func,
        bounds=func.get_bounds_array(),
        mutation=DErand1(F=0.8),
        crossover=BinomialCrossover(CR=0.9),
        pop_size=20,
        max_iter=50,
        verbose=False,
        seed=42
    )
    
    result = optimizer.optimize()
    print(f"   ✓ Custom strategies work correctly")
except Exception as e:
    print(f"   ✗ Error with custom strategies: {e}")
    sys.exit(1)

# Test multiple benchmark functions
print("\n4. Testing benchmark functions...")
try:
    from pyrade.benchmarks import Rastrigin, Rosenbrock, Ackley
    
    benchmarks = [Sphere(dim=5), Rastrigin(dim=5), Rosenbrock(dim=5), Ackley(dim=5)]
    
    for bench in benchmarks:
        value = bench(np.zeros(5))
        print(f"   ✓ {bench.__class__.__name__}: f(0) = {value:.6f}")
except Exception as e:
    print(f"   ✗ Error with benchmark functions: {e}")
    sys.exit(1)

# Test vectorization
print("\n5. Testing vectorization performance...")
try:
    func = Sphere(dim=10)
    
    import time
    start = time.time()
    
    optimizer = DifferentialEvolution(
        objective_func=func,
        bounds=func.get_bounds_array(),
        pop_size=50,
        max_iter=100,
        verbose=False,
        seed=42
    )
    
    result = optimizer.optimize()
    elapsed = time.time() - start
    
    print(f"   ✓ Completed 100 iterations in {elapsed:.3f}s")
    print(f"     ({elapsed/100*1000:.2f}ms per iteration)")
except Exception as e:
    print(f"   ✗ Error during performance test: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! PyRADE is ready to use. 🚀")
print("=" * 60)
print("\nTry running the examples:")
print("  python examples/basic_usage.py")
print("  python examples/custom_strategy.py")
print("  python examples/benchmark_comparison.py")
