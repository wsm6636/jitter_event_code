# Task Chain Latency Analysis Tool

## Overview

This tool implements the methods described in the paper "Investigating Jitter Propagation in Task Chains" 


## Dependencies

```bash
pip install numpy scipy matplotlib pandas
```

## Main Files

- `analysis.py` - Basic analysis algorithm implementation
- `analysisC1.py` - Analysis algorithm with Corollary 1 optimizations
- `main.py` - Main execution file for comparing both algorithms
- `evaluation.py` - Basic algorithm evaluation
- `evaluationC1.py` - Optimized algorithm evaluation
- `plot.py` - Plotting utilities

## Quick Start

### 1. Comparing Both Algorithms

```python
# Experiment parameter configuration
jitters = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]  # Jitter levels: 0% to 50%
num_chains = [3, 5, 8, 10]                           # Task chain lengths: 3 to 10 tasks
num_repeats = 100                                    # Repeat each parameter combination 100 times
random_seed = 1754657734                                  # Fixed random seed for reproducible results
periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000]    # Available task period pool
```

**Parameter Explanation:**

- `jitters`: Test the impact of different jitter levels on algorithm performance
- `num_chains`: Test algorithm performance under different task chain lengths
- `num_repeats`: Increase repetitions to improve statistical reliability
- `random_seed`: Ensure reproducible experimental results
- `periods`: Task periods will be randomly selected from this pool

### 2. Batch Experiments, Visualization, and Data Export

Run complete comparison experiments:

```bash
python main.py
```

This will automatically:

- Perform batch testing on both algorithms
- Generate result CSV files
- Automatically create comparison charts
- Filter and export data by task count
- Output results to `rtssresult/` and `C1/` folders

## Output Results Explanation

### Key Metrics

- **final_e2e_max**: Algorithm-predicted maximum end-to-end latency
- **max_reaction_time**: Actual maximum reaction time obtained through simulation
- **R**: Ratio = max_reaction_time / final_e2e_max
  - R ≤ 1: Algorithm prediction is accurate (safe)
  - R > 1: Algorithm prediction is insufficient (exceeds)
- **false_percentage**: Algorithm failure percentage

### Result Files

```
rtssresult/                    # Basic algorithm results
├── 100_12345_20250814_123456/
│   ├── data_100_12345_20250814_123456.csv     # Original complete data
│   ├── percent_100_12345_20250814_123456.png  # Failure rate charts
│   ├── R_100_12345_20250814_123456.png        # R-value distribution charts
│   └── data/                                  # Filtered data
│       ├── data3.csv                          # All data for 3 tasks
│       ├── data3_20per.csv                    # 3 tasks with 20% jitter data
│       ├── data5.csv                          # All data for 5 tasks
│       ├── data5_20per.csv                    # 5 tasks with 20% jitter data
│       ├── data8.csv                          # All data for 8 tasks
│       ├── data8_20per.csv                    # 8 tasks with 20% jitter data
│       ├── data10.csv                         # All data for 10 tasks
│       └── data10_20per.csv                   # 10 tasks with 20% jitter data

C1/                           # Optimized algorithm results
├── 100_12345_20250814_123456/
│   ├── data_100_12345_20250814_123456.csv     # Original complete data
│   ├── percent_100_12345_20250814_123456.png  # Failure rate charts
│   ├── R_100_12345_20250814_123456.png        # R-value distribution charts
│   └── data/                                  # Filtered data
│       ├── data3.csv                          # (same structure as above)
│       ├── data3_20per.csv
│       └── ...

compare/                      # Comparison results
├── 100_12345_20250814_123456/
│   ├── compare_percent_100_12345_20250814_123456.png  # Failure rate comparison
│   └── compare_R_100_12345_20250814_123456.png        # R-value distribution comparison

log/                          # Detailed logs
├── rtssresult_log_100_12345_20250814_123456.txt      # Basic algorithm log
└── C1_log_100_12345_20250814_123456.txt              # Optimized algorithm log
```

**File Naming Convention:** `{num_repeats}_{random_seed}_{timestamp}`

- `num_repeats`: Number of experiment repetitions
- `random_seed`: Random seed
- `timestamp`: Experiment execution timestamp
