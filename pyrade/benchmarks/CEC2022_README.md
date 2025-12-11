# CEC 2022 Benchmark Data

The CEC 2022 benchmark suite requires input data files (shift vectors, rotation matrices, and shuffle data).

## Required Data Files

For each function and dimension, you need:

### Single Functions (F1-F5):
- `M_{func_num}_D{dim}.txt` - Rotation matrix
- `shift_data_{func_num}.txt` - Shift vector

### Hybrid Functions (F6-F8):
- `M_{func_num}_D{dim}.txt` - Rotation matrices
- `shift_data_{func_num}.txt` - Shift vectors
- `shuffle_data_{func_num}_D{dim}.txt` - Shuffle indices

### Composition Functions (F9-F12):
- `M_{func_num}_D{dim}.txt` - Multiple rotation matrices
- `shift_data_{func_num}.txt` - Multiple shift vectors

## Directory Structure

Place all data files in a directory (default: `pyrade/benchmarks/cec2022_data/`):

```
pyrade/benchmarks/cec2022_data/
├── M_1_D2.txt
├── M_1_D10.txt
├── M_1_D20.txt
├── shift_data_1.txt
├── M_2_D2.txt
├── M_2_D10.txt
...
```

## Data File Format

### Rotation Matrix (`M_*.txt`)
- Plain text file with space-separated values
- For single functions: dim × dim matrix
- For composition functions: (cf_num × dim) × dim matrix

### Shift Data (`shift_data_*.txt`)
- Plain text file with space-separated values
- For single functions: 1 row with dim values
- For composition functions: cf_num rows with dim values each

### Shuffle Data (`shuffle_data_*.txt`)
- Plain text file with integer indices (1-indexed)
- One row with dim values

## Obtaining the Data Files

The official CEC 2022 data files can be obtained from:
- CEC 2022 Competition Website
- Or contact the competition organizers

**Note**: The data files are NOT included with PyRADE due to licensing restrictions. You must obtain them separately from the official source.

## Usage

```python
from pyrade.benchmarks import CEC2022

# Initialize with custom data directory
func = CEC2022(func_num=1, dim=10, data_dir='path/to/cec2022_data')

# Evaluate
x = np.random.uniform(-100, 100, 10)
fitness = func(x)
```

## Dimensions

Functions are defined for:
- D = 2, 10, 20

**Note**: Functions F6, F7, F8 are NOT defined for D=2.

## Function List

1. F1: Zakharov (300)
2. F2: Rosenbrock (400)
3. F3: Schaffer F7 (600)
4. F4: Step Rastrigin (800)
5. F5: Levy (900)
6. F6: Hybrid 1 (1800) - D ≥ 10
7. F7: Hybrid 2 (2000) - D ≥ 10
8. F8: Hybrid 3 (2200) - D ≥ 10
9. F9: Composition 1 (2300)
10. F10: Composition 2 (2400)
11. F11: Composition 3 (2600)
12. F12: Composition 4 (2700)

Numbers in parentheses are the bias values added to function values.
