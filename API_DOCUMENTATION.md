# PyRADE API Documentation

## Table of Contents
1. [Core Classes](#core-classes)
2. [Mutation Strategies](#mutation-strategies)
3. [Crossover Strategies](#crossover-strategies)
4. [Selection Strategies](#selection-strategies)
5. [Benchmark Functions](#benchmark-functions)
6. [Utility Classes](#utility-classes)

---

## Core Classes

### DifferentialEvolution

Main optimizer class for Differential Evolution.

#### Constructor

```python
DifferentialEvolution(
    objective_func,      # Callable: function to minimize
    bounds,              # tuple or list: search space bounds
    mutation=None,       # MutationStrategy: mutation operator
    crossover=None,      # CrossoverStrategy: crossover operator
    selection=None,      # SelectionStrategy: selection operator
    pop_size=50,         # int: population size
    max_iter=1000,       # int: maximum iterations
    seed=None,           # int: random seed
    verbose=False,       # bool: print progress
    callback=None        # callable: progress callback
)
```

#### Methods

**optimize()**
- Returns: `dict` with keys:
  - `'best_solution'`: Best solution found (ndarray)
  - `'best_fitness'`: Best fitness value (float)
  - `'n_iterations'`: Number of iterations (int)
  - `'history'`: Optimization history (dict)
  - `'success'`: Success flag (bool)
  - `'time'`: Total time (float)

#### Attributes

- `best_solution_`: Best solution found (ndarray)
- `best_fitness_`: Best fitness value (float)
- `history_`: Optimization history (dict)

#### Example

```python
from pyrade import DifferentialEvolution

def my_func(x):
    return sum(x**2)

optimizer = DifferentialEvolution(
    objective_func=my_func,
    bounds=[(-10, 10)] * 5,
    pop_size=50,
    max_iter=200
)

result = optimizer.optimize()
print(f"Best: {result['best_fitness']}")
```

---

### Population

Population management class (used internally).

#### Constructor

```python
Population(
    pop_size,    # int: population size
    dim,         # int: problem dimensionality
    bounds,      # tuple or list: bounds
    seed=None    # int: random seed
)
```

#### Methods

**initialize_random()**
- Initializes population using Latin Hypercube Sampling

**evaluate(objective_func)**
- Evaluates fitness for all individuals
- Returns: fitness array

**update(new_vectors, new_fitness)**
- Updates population with new vectors and fitness

---

## Mutation Strategies

All mutation strategies inherit from `MutationStrategy` base class.

### DErand1

Most common mutation: `v = x_r1 + F * (x_r2 - x_r3)`

```python
from pyrade.operators import DErand1

mutation = DErand1(F=0.8)
```

**Parameters:**
- `F` (float): Mutation factor, range [0, 2], default=0.8

**Characteristics:**
- Good balance between exploration and exploitation
- Most widely used strategy
- Works well on most problems

---

### DEbest1

Exploitative mutation: `v = x_best + F * (x_r1 - x_r2)`

```python
from pyrade.operators import DEbest1

mutation = DEbest1(F=0.8)
```

**Parameters:**
- `F` (float): Mutation factor, range [0, 2], default=0.8

**Characteristics:**
- Fast convergence
- Good for unimodal functions
- May get stuck in local optima

---

### DEcurrentToBest1

Balanced mutation: `v = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)`

```python
from pyrade.operators import DEcurrentToBest1

mutation = DEcurrentToBest1(F=0.8)
```

**Parameters:**
- `F` (float): Mutation factor, range [0, 2], default=0.8

**Characteristics:**
- Balances exploration and exploitation
- Often performs well across different problems
- Good general-purpose choice

---

### DErand2

Exploratory mutation: `v = x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - x_r5)`

```python
from pyrade.operators import DErand2

mutation = DErand2(F=0.8)
```

**Parameters:**
- `F` (float): Mutation factor, range [0, 2], default=0.8

**Characteristics:**
- More exploratory
- Uses two difference vectors
- Good for highly multimodal problems

---

## Crossover Strategies

All crossover strategies inherit from `CrossoverStrategy` base class.

### BinomialCrossover

Standard binomial crossover (most common).

```python
from pyrade.operators import BinomialCrossover

crossover = BinomialCrossover(CR=0.9)
```

**Parameters:**
- `CR` (float): Crossover probability, range [0, 1], default=0.9

**Characteristics:**
- Independent dimension crossover
- Most commonly used
- Higher CR = more from mutant

---

### ExponentialCrossover

Copies contiguous segment from mutant.

```python
from pyrade.operators import ExponentialCrossover

crossover = ExponentialCrossover(CR=0.9)
```

**Parameters:**
- `CR` (float): Crossover probability, range [0, 1], default=0.9

**Characteristics:**
- Preserves building blocks
- Segment length follows geometric distribution
- Alternative to binomial

---

### UniformCrossover

Equal probability crossover (CR=0.5).

```python
from pyrade.operators import UniformCrossover

crossover = UniformCrossover()
```

**Characteristics:**
- 50% probability per dimension
- Simple and unbiased
- No parameters needed

---

## Selection Strategies

All selection strategies inherit from `SelectionStrategy` base class.

### GreedySelection

Standard greedy selection (most common).

```python
from pyrade.operators import GreedySelection

selection = GreedySelection()
```

**Characteristics:**
- Keep trial if better, else keep target
- Monotonic improvement
- Standard DE selection

---

### TournamentSelection

Tournament-based selection.

```python
from pyrade.operators import TournamentSelection

selection = TournamentSelection(tournament_size=2)
```

**Parameters:**
- `tournament_size` (int): Tournament size, default=2

**Characteristics:**
- Selects best from random subset
- Can add diversity
- Larger size = more pressure

---

### ElitistSelection

Guarantees elite preservation.

```python
from pyrade.operators import ElitistSelection

selection = ElitistSelection(elite_size=1)
```

**Parameters:**
- `elite_size` (int): Number of elites to preserve, default=1

**Characteristics:**
- Guarantees best individuals survive
- Prevents loss of good solutions
- Useful for difficult problems

---

## Benchmark Functions

All benchmark functions inherit from `BenchmarkFunction` base class.

### Sphere

Simple unimodal function: `f(x) = sum(x^2)`

```python
from pyrade.benchmarks import Sphere

func = Sphere(dim=30)
```

- **Domain:** [-100, 100]^d
- **Global minimum:** f(0, ..., 0) = 0

---

### Rastrigin

Highly multimodal: `f(x) = 10*d + sum(x^2 - 10*cos(2*pi*x))`

```python
from pyrade.benchmarks import Rastrigin

func = Rastrigin(dim=30)
```

- **Domain:** [-5.12, 5.12]^d
- **Global minimum:** f(0, ..., 0) = 0

---

### Rosenbrock

Valley-shaped function.

```python
from pyrade.benchmarks import Rosenbrock

func = Rosenbrock(dim=30)
```

- **Domain:** [-5, 10]^d
- **Global minimum:** f(1, ..., 1) = 0

---

### Ackley

Many local minima.

```python
from pyrade.benchmarks import Ackley

func = Ackley(dim=30)
```

- **Domain:** [-32, 32]^d
- **Global minimum:** f(0, ..., 0) = 0

---

### Additional Functions

- **Griewank**: Multimodal with widespread local minima
- **Schwefel**: Deceptive multimodal
- **Levy**: Multimodal
- **Michalewicz**: Steep valleys
- **Zakharov**: Unimodal

All have the same usage pattern:
```python
from pyrade.benchmarks import FunctionName

func = FunctionName(dim=30)
value = func(x)  # Evaluate at point x
bounds = func.get_bounds_array()  # Get bounds for optimizer
optimum = func.optimum  # Known global optimum
```

---

## Utility Classes

### Boundary Handlers

#### ClipBoundary

Clip to bounds (most common).

```python
from pyrade.utils import ClipBoundary

handler = ClipBoundary()
repaired = handler.repair(vectors, lb, ub)
```

#### ReflectBoundary

Reflect at boundaries.

```python
from pyrade.utils import ReflectBoundary

handler = ReflectBoundary()
```

#### RandomBoundary

Replace with random values.

```python
from pyrade.utils import RandomBoundary

handler = RandomBoundary()
```

#### WrapBoundary

Toroidal topology.

```python
from pyrade.utils import WrapBoundary

handler = WrapBoundary()
```

---

### Termination Criteria

#### MaxIterations

Terminate after max iterations.

```python
from pyrade.utils import MaxIterations

criterion = MaxIterations(max_iter=1000)
```

#### FitnessThreshold

Terminate when fitness reaches threshold.

```python
from pyrade.utils import FitnessThreshold

criterion = FitnessThreshold(threshold=1e-6)
```

#### NoImprovement

Terminate if no improvement.

```python
from pyrade.utils import NoImprovement

criterion = NoImprovement(patience=50, min_delta=1e-6)
```

---

## Creating Custom Strategies

### Custom Mutation

```python
from pyrade.operators import MutationStrategy
import numpy as np

class MyMutation(MutationStrategy):
    def __init__(self, F=0.8):
        self.F = F
    
    def apply(self, population, fitness, best_idx, target_indices):
        # Implement vectorized mutation
        # Return mutants of shape (pop_size, dim)
        mutants = ...
        return mutants
```

### Custom Crossover

```python
from pyrade.operators import CrossoverStrategy
import numpy as np

class MyCrossover(CrossoverStrategy):
    def __init__(self, CR=0.9):
        self.CR = CR
    
    def apply(self, population, mutants):
        # Implement vectorized crossover
        # Return trials of shape (pop_size, dim)
        trials = ...
        return trials
```

### Custom Selection

```python
from pyrade.operators import SelectionStrategy
import numpy as np

class MySelection(SelectionStrategy):
    def apply(self, population, fitness, trials, trial_fitness):
        # Implement selection logic
        # Return (new_population, new_fitness)
        new_population = ...
        new_fitness = ...
        return new_population, new_fitness
```

---

## Performance Tips

1. **Always use vectorized NumPy operations**
2. **Typical pop_size: 5-10x problem dimension**
3. **Standard parameters: F=0.8, CR=0.9**
4. **Choose mutation based on problem:**
   - Unimodal → DE/best/1
   - Multimodal → DE/rand/1 or DE/rand/2
   - Unknown → DE/current-to-best/1

---

For more examples, see the `examples/` directory!
