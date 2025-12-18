# Paper Results Summary - Original DE (DErand1/bin)

## Experimental Configuration

**Algorithm:** Original Differential Evolution (DErand1/bin)
- **Mutation Strategy:** DE/rand/1 (Classic DE mutation)
- **Crossover Strategy:** Binomial crossover
- **Selection Strategy:** Elitist (greedy) selection

**Parameters:**
- Population Size: 50
- Mutation Factor (F): 0.8
- Crossover Rate (CR): 0.9
- Max Generations: 500
- Independent Runs: 30 per benchmark

**Problem Settings:**
- Dimension: 30D for all benchmarks
- Total Function Evaluations: 25,000 per run (50 × 500)

## Benchmark Functions Tested

| Function   | Domain              | Characteristics                          | Global Optimum |
|-----------|---------------------|------------------------------------------|----------------|
| Sphere     | [-100, 100]³⁰       | Unimodal, smooth, convex                | f(0,...,0) = 0 |
| Rastrigin  | [-5.12, 5.12]³⁰     | Multimodal, highly rugged               | f(0,...,0) = 0 |
| Rosenbrock | [-5, 10]³⁰          | Multimodal, narrow valley               | f(1,...,1) = 0 |
| Ackley     | [-32.768, 32.768]³⁰ | Multimodal, many local optima           | f(0,...,0) = 0 |
| Griewank   | [-600, 600]³⁰       | Multimodal, product-sum composition     | f(0,...,0) = 0 |
| Schwefel   | [-500, 500]³⁰       | Highly multimodal, deceptive            | f(420.97,...) = 0 |
| Zakharov   | [-5, 10]³⁰          | Unimodal, similar to sphere             | f(0,...,0) = 0 |

## Generated Visualizations

Each benchmark folder contains:

### 1. convergence_curve.png
- **Content:** Mean convergence curve with ±1 std deviation bands
- **Y-axis:** Log scale fitness values
- **Shows:** Individual runs (light blue), mean trajectory (red), variability bands
- **Purpose:** Analyze convergence speed and consistency

### 2. fitness_distribution.png
- **Content:** Box plot of final fitness values
- **Statistics:** Mean, Median, Std Dev, Min, Max displayed
- **Shows:** Distribution spread, outliers, central tendency
- **Purpose:** Statistical performance summary

### 3. fitness_histogram.png
- **Content:** Histogram of final best fitness values
- **Features:** Mean (red line), Median (green line)
- **Shows:** Distribution shape, skewness
- **Purpose:** Frequency analysis of final solutions

### 4. landscape_2d.png
- **Content:** 2D contour plot (first two dimensions)
- **Features:** Function landscape + best solutions from all runs (red stars)
- **Shows:** Search space characteristics, solution clustering
- **Purpose:** Visual understanding of optimization landscape

### 5. statistical_summary.png
- **Content:** 4-panel comprehensive summary
  - Panel 1: Mean convergence with error bands
  - Panel 2: Fitness distribution box plot
  - Panel 3: Success rate curves (multiple thresholds)
  - Panel 4: Detailed statistics table
- **Purpose:** Complete single-figure summary for papers

### 6. numerical_results.csv
- **Content:** Raw numerical data from all 30 runs
- **Format:** CSV with run number and final fitness
- **Statistics:** Mean, Median, Std, Min, Max included
- **Purpose:** Further analysis, statistical tests, comparison tables

## Overall Comparison

### overall_comparison.png
- **Content:** Cross-benchmark performance comparison
  - Left panel: Bar chart with error bars (log scale)
  - Right panel: Box plots for all benchmarks
- **Purpose:** Compare DE performance across different problem types

## Directory Structure

```
paper_result/
├── sphere/
│   ├── convergence_curve.png
│   ├── fitness_distribution.png
│   ├── fitness_histogram.png
│   ├── landscape_2d.png
│   ├── statistical_summary.png
│   └── numerical_results.csv
├── rastrigin/
│   └── [same 6 files]
├── rosenbrock/
│   └── [same 6 files]
├── ackley/
│   └── [same 6 files]
├── griewank/
│   └── [same 6 files]
├── schwefel/
│   └── [same 6 files]
├── zakharov/
│   └── [same 6 files]
├── overall_comparison.png
└── RESULTS_SUMMARY.md (this file)
```

## Usage Notes

### For Paper Figures
1. **Single benchmark analysis:** Use `statistical_summary.png` - contains all key information
2. **Convergence comparison:** Use `convergence_curve.png` for multiple algorithms side-by-side
3. **Statistical comparison:** Use box plots from `fitness_distribution.png`
4. **Cross-benchmark overview:** Use `overall_comparison.png`

### For Tables
Use data from `numerical_results.csv` files to create comparison tables with:
- Mean ± Std Dev
- Median values
- Best/Worst performance
- Statistical significance tests (t-test, Wilcoxon)

### For Detailed Analysis
- Check convergence speed from `convergence_curve.png`
- Analyze solution quality spread from histograms
- Verify landscape exploration from `landscape_2d.png`
- Compare success rates at different thresholds from `statistical_summary.png` Panel 3

## Key Observations

### Expected Performance Patterns
1. **Sphere & Zakharov** (Unimodal): Should show smooth convergence, low variance
2. **Rastrigin & Schwefel** (Highly multimodal): Challenging, higher variance expected
3. **Rosenbrock** (Valley): Slow convergence due to narrow valley structure
4. **Ackley & Griewank** (Multimodal): Moderate difficulty, good for DE

### Quality Indicators
- **Low std deviation:** Consistent performance across runs
- **Smooth convergence:** Effective parameter settings
- **No premature convergence:** Good exploration-exploitation balance

## Reproducibility

To regenerate these results:
```bash
python generate_paper_results.py
```

**Note:** Results are stochastic. Random seeds (0-29) ensure reproducibility of exact values.

## Citation Format (Example)

"We evaluated the original DE algorithm (DErand1/bin) with F=0.8, CR=0.9, and population size 50
on seven benchmark functions in 30 dimensions over 30 independent runs of 500 generations each."

---

Generated: December 2024
Algorithm: PyRADE v0.4.4
Configuration: Original DE (DErand1/bin)
Total Runs: 210 (7 benchmarks × 30 runs each)
