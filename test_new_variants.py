"""Quick test of new classic DE variants."""
import numpy as np
from pyrade.algorithms.classic import DEbest2bin, DEcurrentToRand1bin, DERandToBest1bin

def sphere(x):
    return np.sum(x**2)

print("Testing new classic variants...")

# Test DEbest2bin
de1 = DEbest2bin(sphere, bounds=[(-10,10)]*5, pop_size=20, max_iter=50, verbose=False)
r1 = de1.optimize()
print(f"✓ DEbest2bin: {r1['best_fitness']:.6e}")

# Test DEcurrentToRand1bin
de2 = DEcurrentToRand1bin(sphere, bounds=[(-10,10)]*5, pop_size=20, max_iter=50, verbose=False)
r2 = de2.optimize()
print(f"✓ DEcurrentToRand1bin: {r2['best_fitness']:.6e}")

# Test DERandToBest1bin
de3 = DERandToBest1bin(sphere, bounds=[(-10,10)]*5, pop_size=20, max_iter=50, verbose=False)
r3 = de3.optimize()
print(f"✓ DERandToBest1bin: {r3['best_fitness']:.6e}")

print("\n✓ All 3 new classic variants working!")
