# Task Chain Latency Analysis Tool

## Overview

This tool implements the methods described in the paper "Investigating Jitter Propagation in Task Chains"

## Dependencies

```bash
pip install numpy scipy matplotlib pandas
```

## Main Files

- `analysis.py` - Basic analysis algorithm implementation
- `analysisadjust.py` - Analysis algorithm with Corollary 1 optimizations
- `main.py` - Main execution file for comparing both algorithms
- `evaluation.py` - Basic algorithm evaluation
- `evaluationadjust.py` - Optimized algorithm evaluation
- `plot.py` - Plotting utilities
- `generate_comparison.py` - Comparison result graph and filtered data file
- `run_experiments.sh` - Experiment script

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
chmod +x run.sh
./run.sh 100 2             # 2(NUM_EXPERIMENTS) experiments were performed in parallel, with 100(NUM_REPEATS) replicates per experiment.
```

NUM_REPEATS: Number of repetitions per experiment (default 100)
NUM_EXPERIMENTS: Number of concurrent experiments (default 5)

## Output Results Explanation

### Key Metrics

- **final_e2e_max**: Algorithm-predicted maximum end-to-end latency
- **max_reaction_time**: Actual maximum reaction time obtained through simulation
- **R**: Ratio = max_reaction_time / final_e2e_max
  - R ≤ 1: Algorithm prediction is accurate (safe)
  - R > 1: Algorithm prediction is insufficient (exceeds)
- **false_percentage**: Algorithm failure percentage

### Result Filecompare

```
rtssresult/
└── data_100_12345_20250814_123456.csv     # Original data

adjust/                                                                           # Optimized algorithm results
└── data_100_12345_20250814_123456.csv

compare/
└── 20250901_143012_123/
  ├── common_results_20250901_143012_123.csv          # RTSS
  ├── common_results_adjust_20250901_143012_123.csv       # adjust
  ├── final_compare_percent.png
  ├── final_compare_histogram.png
  └── data/
    ├──data3.csv …
    └── adjust_data3.csv …

log/                          # Detailed logs
├── rtssresult_log_100_12345_20250814_123456.txt      # Basic algorithm log
└── adjust_log_100_12345_20250814_123456.txt              # Optimized algorithm log
```
