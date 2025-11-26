# PyRADE v0.2.0 - Visualization & Experiment Management

## 🎉 Major New Features

### 📊 ExperimentManager Class
High-level interface for running and managing benchmark experiments with automatic visualization and data export.

```python
from pyrade import ExperimentManager

exp = ExperimentManager(
    benchmarks=['Sphere', 'Rastrigin', 'Rosenbrock'],
    dimensions=10,
    n_runs=30
)
exp.run_complete_pipeline()  # Experiments + Plots + Exports + Report
```

**Features:**
- ✅ 11 built-in benchmark functions + custom function support
- ✅ Automatic visualization generation (convergence, boxplots)
- ✅ Multi-format export (CSV, NumPy, JSON)
- ✅ Timestamped experiment folders
- ✅ Statistical analysis and reports

### 🎨 Comprehensive Visualization Module
Professional plotting for optimization analysis with 11 plot types:

- Convergence curves (with std deviation)
- Fitness distribution boxplots
- 2D/3D Pareto fronts
- Parameter heatmaps
- Parallel coordinate plots
- Contour landscapes with trajectories
- Hypervolume & IGD progress
- Population diversity tracking

```python
from pyrade import OptimizationVisualizer

viz = OptimizationVisualizer()
viz.plot_convergence_curve(history, log_scale=True, save_path="plot.png")
```

## 📦 What's Included

**New Classes:**
- `ExperimentManager` - Complete experiment pipeline
- `OptimizationVisualizer` - 11 professional plot types

**New Functions:**
- `calculate_hypervolume_2d()` - Multi-objective metric
- `calculate_igd()` - Inverted Generational Distance
- `is_pareto_efficient()` - Pareto front identification

**Output Structure:**
```
experiment_2025-11-27_12-30-45/
├── convergence_plots/       # Individual plots per benchmark
├── fitness_boxplot_all.png  # Distribution comparison
├── csv_exports/             # Summary + detailed data
├── numpy_data/              # Raw arrays (.npy)
├── summary_results.json     # Structured results
└── experiment_report.txt    # Human-readable report
```

## 🚀 Quick Start

```bash
pip install pyrade==0.2.0
```

```python
# Example 1: Quick benchmark test
from pyrade import ExperimentManager

exp = ExperimentManager(
    benchmarks=['Sphere', 'Rastrigin', 'Rosenbrock'],
    dimensions=10,
    n_runs=30
)
exp.run_complete_pipeline()
```

```python
# Example 2: Custom experiment
exp = ExperimentManager(
    benchmarks=['Ackley', 'Griewank'],
    dimensions=20,
    n_runs=50,
    population_size=100,
    max_iterations=200,
    experiment_name='my_experiment',
    seed=42
)
exp.run_experiments()
exp.plot_all()
exp.export_results()
```

## 📚 Available Benchmarks

11 built-in functions: Sphere, Rastrigin, Rosenbrock, Ackley, Griewank, Schwefel, Levy, Michalewicz, Zakharov, Easom, StyblinskiTang

## 🔧 Requirements

- Python >= 3.7
- numpy >= 1.20.0
- matplotlib >= 3.3.0
- pandas >= 1.2.0

## 📖 Documentation

- Full release notes: [RELEASE_v0.2.0.md](RELEASE_v0.2.0.md)
- ExperimentManager guide: [docs/experiment_manager.md](docs/experiment_manager.md)
- Examples: [examples/experiment_manager_demo.py](examples/experiment_manager_demo.py)

## ⚙️ Migration from v0.1.0

**No breaking changes!** All v0.1.0 code works in v0.2.0.

v0.2.0 adds high-level interfaces while maintaining full backward compatibility.

## 🎯 Use Cases

- **Research**: Automated benchmarking with publication-ready plots
- **Industry**: Performance testing with comprehensive reporting
- **Education**: Visual demonstration of optimization concepts

## 📊 Example Output

Running `ExperimentManager` generates:
- ✅ Individual convergence plots for each benchmark
- ✅ Combined comparison plots
- ✅ Boxplots for performance comparison
- ✅ CSV files with summary statistics and detailed data
- ✅ NumPy arrays for custom analysis
- ✅ JSON structured results
- ✅ Text report with rankings

## 🐛 Bug Fixes

- Fixed import issues in benchmark modules
- Improved error handling in visualization
- Better memory management for large experiments

## 🔮 Coming Next (v0.3.0)

- Parallel experiment execution
- Multi-objective DE algorithms
- Real-time progress visualization
- Statistical significance testing

---

**Full Changelog**: v0.1.0...v0.2.0
**Install**: `pip install pyrade==0.2.0`
**Docs**: https://github.com/arartawil/pyrade
