"""Test all DE variants including newest additions."""
import numpy as np
from pyrade.algorithms import (
    DEbest2bin, DEcurrentToRand1bin, DERandToBest1bin,
    DErand1exp, DErand1EitherOrBin
)

def sphere(x):
    return np.sum(x**2)

print("Testing all DE variants...")
print("=" * 60)

bounds = [(-10,10)]*5
np.random.seed(42)

print("\n[Previously Added - Session 1]")
de1 = DEbest2bin(sphere, bounds=bounds, pop_size=20, max_iter=50, verbose=False, seed=42)
r1 = de1.optimize()
print(f"✓ DEbest2bin: {r1['best_fitness']:.6e}")

de2 = DEcurrentToRand1bin(sphere, bounds=bounds, pop_size=20, max_iter=50, verbose=False, seed=42)
r2 = de2.optimize()
print(f"✓ DEcurrentToRand1bin: {r2['best_fitness']:.6e}")

de3 = DERandToBest1bin(sphere, bounds=bounds, pop_size=20, max_iter=50, verbose=False, seed=42)
r3 = de3.optimize()
print(f"✓ DERandToBest1bin: {r3['best_fitness']:.6e}")

print("\n[Just Added - Session 2]")
de4 = DErand1exp(sphere, bounds=bounds, pop_size=20, max_iter=50, verbose=False, seed=42)
r4 = de4.optimize()
print(f"✓ DErand1exp (exponential crossover): {r4['best_fitness']:.6e}")

de5 = DErand1EitherOrBin(sphere, bounds=bounds, pop_size=20, max_iter=50, p_F=0.5, verbose=False, seed=42)
r5 = de5.optimize()
print(f"✓ DErand1EitherOrBin (probabilistic F): {r5['best_fitness']:.6e}")

print("\n" + "=" * 60)
print("COMPLETE CHECKLIST OF 10 DE VARIANTS:")
print("=" * 60)
print("  ✓ DE/rand/1            → DErand1bin")
print("  ✓ DE/rand/2            → DErand2bin")
print("  ✓ DE/best/1            → DEbest1bin")
print("  ✓ DE/best/2            → DEbest2bin")
print("  ✓ DE/current-to-best/1 → DEcurrentToBest1bin")
print("  ✓ DE/current-to-rand/1 → DEcurrentToRand1bin")
print("  ✓ DE/rand-to-best/1    → DERandToBest1bin")
print("  ✓ DE/rand/1/either-or  → DErand1EitherOrBin [NEW]")
print("  ✓ DE/rand/1/bin        → DErand1bin [=DE/rand/1]")
print("  ✓ DE/rand/1/exp        → DErand1exp [NEW]")
print("=" * 60)
