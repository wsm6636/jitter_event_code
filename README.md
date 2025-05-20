# Evaluation

The evaluation of "Understanding Jitter Propagation in Task Chains"

### Requirements

version: python3.12

```
pip install numpy matplotlib scipy seaborn
```

### filetree

1. **analysis.py**: Implementation of Algorithm 2 and the general task chain, and calculation of the maximum reaction time.

2. **evaluation.py**: Set parameters and output ".csv" result file and two plots

    parameters:
  
    - num_repeats: number of repeats

    - periods: periods set

    - jitters: percent jitter set, used by max_jitter = percent_jitter * period

    - num_chains: set of number tasks of per chain

3. **data_{num_repeats}_{timestamp}.csv**

    results_csv: data file with
  
    - R value = max_reaction_time_of_general_task_chain / max_reaction_time_of_Algorithm2
  
    - fales percent: is the percentage of failures of Algorithm 2, which because of FINDEFFECTIVESERIES no return value (Alg2 line1)

4. **percent_{num_repeats}_{timestamp}.png:** Relationship between jitter percentage and failure percentage

5. **R_{num_repeats}_{timestamp}.png:** R_value distribution of different jitter percentage

6. **plot.py**: Read ".csv" file and draw the two plots.

### use

```
python3 evaluation.py
```

If you only need to adjust the plot color, size, etc.

```
python3 plot.py data_xx_xxxx.csv
```
