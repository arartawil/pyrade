# PyRADE v0.2.0 Release Notes

## 🎉 Major Features

### New ExperimentManager Class
Complete high-level interface for benchmark experiments with automatic organization and export.

**Key Features:**
- 🎯 Select from 11 built-in benchmark functions or use custom functions
- 📊 Automatic visualization generation (convergence plots, boxplots)
- 💾 Multi-format data export (CSV, NumPy, JSON)
- 📁 Timestamped experiment folders with organized structure
- 📈 Statistical analysis and ranking reports
- 🔄 Reproducible experiments with seed control

**Quick Example:**
```python
from pyrade import ExperimentManager

exp = ExperimentManager(
    benchmarks=['Sphere', 'Rastrigin', 'Rosenbrock'],
    dimensions=10,
    n_runs=30,
    population_size=50,
    max_iterations=100
)

# Run complete pipeline: experiments + plots + exports + report
exp.run_complete_pipeline()
```

### Comprehensive Visualization Module
Professional plotting capabilities for optimization analysis.

**11 Visualization Types:**
1. **Convergence Curves** - With standard deviation bands
2. **Fitness Boxplots** - Distribution comparisons with auto-scaling
3. **2D Pareto Fronts** - Bi-objective optimization results
4. **3D Pareto Fronts** - Three-objective scatter plots
5. **Parameter Heatmaps** - Population parameter distributions
6. **Parallel Coordinate Plots** - Multi-dimensional exploration
7. **Contour Landscapes** - 2D function landscapes with trajectories
8. **Hypervolume Progress** - Multi-objective quality indicator
9. **IGD Progress** - Inverted Generational Distance tracking
10. **Population Diversity** - Diversity metrics over time
11. **Combined Plots** - Compare multiple algorithms/strategies

**Quick Example:**
```python
from pyrade import OptimizationVisualizer

viz = OptimizationVisualizer()
viz.plot_convergence_curve(
    history=fitness_history,
    log_scale=True,
    show_std=True,
    save_path="convergence.png"
)
```

## 🚀 What's New

### ExperimentManager (`pyrade.ExperimentManager`)
- Automated experiment pipeline for benchmark testing
- Support for all 11 built-in benchmarks + custom functions
- Configurable DE parameters (F, CR, population, iterations)
- Multiple independent runs for statistical significance
- Automatic result organization in timestamped folders

### Visualization Tools (`pyrade.OptimizationVisualizer`)
- 11 professional plot types for optimization analysis
- Publication-quality figures with customizable styling
- Support for multi-objective visualization
- Automatic saving with configurable DPI and formats

### Data Export
- **CSV**: Summary statistics, detailed run data, convergence histories
- **NumPy**: Raw arrays for custom analysis (`.npy` files)
- **JSON**: Structured results for easy integration
- **Text Reports**: Human-readable summaries with rankings

### Helper Functions
- `calculate_hypervolume_2d()` - Hypervolume indicator calculation
- `calculate_igd()` - Inverted Generational Distance metric
- `is_pareto_efficient()` - Pareto front identification

## 📦 Package Structure

```
pyrade/
├── experiments.py          # NEW: ExperimentManager class
├── visualization.py        # NEW: OptimizationVisualizer class
├── __init__.py            # Updated: Export new classes
└── benchmarks/            # All 11 benchmark functions available
```

## 📊 Example Output Structure

When running `ExperimentManager`, creates organized experiment folders:

```
experiment_2025-11-27_12-30-45/
├── config.json                          # Experiment configuration
├── convergence_plots/                   # Individual convergence plots
│   ├── sphere_convergence.png
│   ├── rastrigin_convergence.png
│   └── rosenbrock_convergence.png
├── all_convergence_combined.png        # Combined comparison
├── fitness_boxplot_all.png             # Distribution comparison
├── csv_exports/                         # CSV data
│   ├── summary_statistics.csv
│   ├── sphere_detailed.csv
│   └── convergence/
│       └── sphere_convergence.csv
├── numpy_data/                          # NumPy arrays
│   ├── sphere_convergence.npy
│   ├── sphere_final_fitness.npy
│   └── sphere_best_solutions.npy
├── summary_results.json                 # JSON summary
└── experiment_report.txt                # Text report
```

## 💻 Usage Examples

### Example 1: Quick Benchmark Test
```python
from pyrade import ExperimentManager

exp = ExperimentManager(
    benchmarks=['Sphere', 'Rastrigin', 'Rosenbrock'],
    dimensions=10,
    n_runs=30
)
exp.run_complete_pipeline()
```

### Example 2: Custom Experiment
```python
exp = ExperimentManager(
    benchmarks=['Ackley', 'Griewank', 'Schwefel'],
    dimensions=20,
    n_runs=50,
    population_size=100,
    max_iterations=200,
    F=0.7,
    CR=0.8,
    experiment_name='my_experiment',
    seed=42
)
exp.run_experiments()
exp.plot_all()
exp.export_results()
```

### Example 3: Custom Function
```python
def my_function(x):
    return sum(x**2) + sum(10 * np.sin(x))

exp = ExperimentManager(
    benchmarks=['Sphere', my_function],
    dimensions=10,
    n_runs=20
)
exp.run_complete_pipeline()
```

## 🔧 API Changes

### New Exports in `pyrade.__init__`
```python
from pyrade import (
    DifferentialEvolution,      # Core algorithm
    ExperimentManager,          # NEW: Experiment management
    OptimizationVisualizer,     # NEW: Visualization tools
    calculate_hypervolume_2d,   # NEW: Hypervolume metric
    calculate_igd,              # NEW: IGD metric
    is_pareto_efficient         # NEW: Pareto identification
)
```

## 📚 Documentation

New documentation added:
- `docs/experiment_manager.md` - Complete ExperimentManager guide
- `examples/experiment_manager_demo.py` - 8 usage examples
- `examples/visualization_examples.py` - All 11 plot types demonstrated
- `examples/README.md` - Examples overview

## 🎯 Available Benchmarks

All 11 benchmarks accessible via `ExperimentManager`:
- **Sphere** - Convex unimodal
- **Rastrigin** - Highly multimodal
- **Rosenbrock** - Narrow valley
- **Ackley** - Many local minima
- **Griewank** - Multimodal
- **Schwefel** - Deceptive
- **Levy** - Multimodal
- **Michalewicz** - Steep ridges
- **Zakharov** - Unimodal
- **Easom** - Flat with sharp optimum
- **StyblinskiTang** - Multimodal

## 🔍 Use Cases

**Research & Academia:**
- Benchmark algorithm performance across multiple functions
- Generate publication-ready plots automatically
- Export data for statistical analysis
- Reproducible experiments with seed control

**Industry & Development:**
- Quick performance testing of optimization problems
- Automated experiment reporting
- Integration with existing workflows (CSV/JSON export)
- Visual debugging of algorithm behavior

**Education:**
- Demonstrate optimization concepts with visualizations
- Compare different algorithm configurations
- Analyze convergence behavior
- Understand parameter effects

## ⚙️ Requirements

- Python >= 3.7
- numpy >= 1.20.0
- matplotlib >= 3.3.0
- pandas >= 1.2.0 (for CSV export)

## 📥 Installation

```bash
# From PyPI (once published)
pip install pyrade==0.2.0

# From source
git clone https://github.com/arartawil/pyrade.git
cd pyrade
git checkout v0.2.0
pip install -e .
```

## 🐛 Bug Fixes

- Fixed import issues in benchmark modules
- Improved error handling in visualization
- Better memory management for large experiments

## 🚦 Migration from v0.1.0

No breaking changes! All v0.1.0 code remains compatible.

**New in v0.2.0:**
```python
# Old way (still works)
from pyrade import DifferentialEvolution
de = DifferentialEvolution(objective_func=func, bounds=bounds)
result = de.optimize()

# New way (high-level interface)
from pyrade import ExperimentManager
exp = ExperimentManager(benchmarks=['Sphere'])
exp.run_complete_pipeline()
```

## 📈 Performance

- ExperimentManager adds minimal overhead (<1%)
- Visualization generation is fast (< 1 second per plot)
- Memory efficient: processes benchmarks sequentially
- Parallel execution support planned for v0.3.0

## 🎓 Examples

Try the included examples:
```bash
cd examples
python experiment_manager_demo.py      # 8 usage examples
python visualization_examples.py       # 11 plot types
```

## 🙏 Acknowledgments

Thanks to the optimization research community for feedback and suggestions.

## 📞 Support

- **Documentation**: https://pyrade.readthedocs.io
- **Issues**: https://github.com/arartawil/pyrade/issues
- **Discussions**: https://github.com/arartawil/pyrade/discussions

## 🔮 Coming in v0.3.0

- Parallel experiment execution
- Multi-objective DE algorithms
- Real-time progress visualization
- Experiment comparison tools
- More benchmark functions
- Statistical significance testing

---

**Full Changelog**: https://github.com/arartawil/pyrade/compare/v0.1.0...v0.2.0
