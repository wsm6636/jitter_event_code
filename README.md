# Evaluation

The evaluation of "Understanding Jitter Propagation in Task Chains"

### Requirements

version: python3.12

scipy: import BasinHoping, used to calculate the global maximum reaction time

numpy, matplotlib: read the csv file storing the data and draw the graph


```
pip install scipy numpy matplotlib
```

### filetree

1. **analysis.py:** Implementation of Algorithm 2 and the general task chain, and calculation of the maximum reaction time.
2. **evaluation.py:** Set parameters and output ".csv" result file and two plots.

   parameters:

   - num_repeats: number of repeats
   - periods: periods set
   - jitter: percent jitter set, used by max_jitter = percent_jitter * period
   - num_chains: set of number tasks of per chain
   - min_period: minimum period for "ratio"
   - ratios: set of ratio
   - random_seed: Random seed, default is current time. Used to trace back a certain experiment
3. **analysisF16.py, evaluationF16.py:** During the algorithm operation, adjust the offsets so that the chain conforms to the Formula (16) in the RTSS`25 paper.
4. **analysisratio.py, evaluationratio.py:** Randomly generate period and offset sets that matches the $r$ value. $r =  \frac{T_{i+1}}{T_{i}}$
5. **plot.py:** Read ".csv" file and draw the plots.
6. **main.py:** Run all experiments and generate comparison results between RTSS and F16

#### output files:

1. path: rtssresult and F16 and ratio: csv and png result files of the corresponding experiments. compare: Comparative experiment results. log: All generated tasks information for each experiment.
2. **{path}/{num_repeats}\_{randomseed}/data_{num_repeats}\_{randomseed}_{timestamp}.csv**

   results_csv: data file with

   - R value = max_reaction_time_of_general_task_chain / max_reaction_time_of_Algorithm2
   - fales percent: is the percentage of failures of Algorithm 2, which because of FINDEFFECTIVESERIES no return value (Alg2 line1)

   **{path}/{num_repeats}\_{randomseed}/percent_{num_repeats}\_{randomseed}_{timestamp}.png:** Relationship between jitter percentage and failure percentage.

   **{path}/{num_repeats}\_{randomseed}/R_{num_repeats}\_{randomseed}_{timestamp}.png:** R_value distribution of different jitter percentage.

   **{path}/{num_repeats}\_{randomseed}/ratios_{num_repeats}\_{randomseed}_{timestamp}.png:**  The impact of ratio on failure rate.

   **{path}/{num_repeats}\_{randomseed}/compare_R_{num_repeats}\_{randomseed}_{timestamp}.png, compare_percent_{num_repeats}\_{randomseed}_{timestamp}.png:** Comparative experiment results.

   **{path}/{num_repeats}\_{randomseed}/xxx_log_{num_repeats}\_{randomseed}_{timestamp}.txt:** tasks information.

### use

**run**

```
python3 main.py
```

**Modify parameters**

```
# main.py line 101~114
    num_repeats = 100  
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000] 
    jitters = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5] 
    num_chains = [3,5,8,10] 
    min_period = 1 
    ratios = np.arange(1.0, 6.0, 0.5)
    random_seed = int(time.time())
```

**If you only need to adjust the plot color, size, etc.**

```
python3 plot.py path/xx_xxxx/data_xx_xxxx.csv path/xx_xxxx/xxx_xx_xxxx.png
```
# event_jitter_demo
