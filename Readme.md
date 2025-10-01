# Jitter Propagation in Task Chains

The repository is used to reproduce the evaluation from

*"Jitter Propagation in Task Chains". Shumo Wang, Enrico Bini, Qingxu Deng, Martina Maggio  46th IEEE Real-Time Systems Symposium (RTSS), Boston, MA, USA, 2025*

The tool supports both **passive** and **active** jitter analysis for cause-effect chains, and supports two mainstream models: LET (Logical Execution Time) and IC (Implicit Communication).

To simulate the task scheduling  maximum reaction time analysis, we borrowed some code from paper 20. In our paper, we set it as $D^{FF}_{base}$.

> [20] M. G¨unzel, K.-H. Chen, N. Ueter, G. von der Br¨uggen, M. D¨urr, and J.-J. Chen, “Timing analysis of asynchronized distributed cause-effect chains,” in Real Time and Embedded Technology and Applications Symposium (RTAS), 2021.

This document is organized as follows:

1. [Environment Setup](#environment-setup)
2. [How to run the experiments](#how-to-run-the-experiments)
3. [Overview of the corresponding functions](#overview-of-the-corresponding-functions)
4. [Miscellaneous](#miscellaneous)

## Environment Setup

### Requirements

To run the experiments Python 3.12 is required. Moreover, the following packages are
required:

```
pip install numpy scipy matplotlib pandas
```

In case there is any dependent package missing, please install them accordingly.

### File Structure

```
├── active                       # Active experiment results
├── compare                      # Compare experiment results
├── passive                      # Passive experiment results
├── utilities                    # Copied from [20]
│   ├── analyzer.py              # Methods to analyze end-to-end timing behavior
│   ├── analyzer_our.py          # Methods to analyze mrt, let
│   ├── augmented_job_chain.py   # Augmented job chains as in the paper
│   ├── chain.py                 # Cause-effect chains
│   ├── evaluation.py            # Methods to draw plots
│   ├── event_simulator.py       # Event-driven simulator with fixed execution time
│   ├── task.py                  # Tasks
├── analysis_active.py           # Active analysis implementation
├── analysis_Gunzel.py           # Günzel LET & IC wrappers
├── analysis_passive.py          # Passive analysis implementation
├── evaluation_active.py         # Active experiment runner
├── evaluation_passive.py        # Passive experiment runner
├── generate_comparison.py       # Plot & CSV post-processing
├── main.py                      # Main function
├── plot.py                      # Visualization utilities
├── README.md
└── run_experiments.sh           # Batch experiment script
```

### Deployment

The following steps explain how to deploy this framework on the machine:

First, clone the git repository or download
the [zip file](https://github.com/wsm6636/jitter_event_code/archive/refs/heads/main.zip):

```
git clone https://github.com/wsm6636/jitter_event_code.git
```

### Quick Start

Move into the code folder and execute run_experiments.sh natively:

```
cd jitter_event_code
bash run_experiments.sh [NUM_REPEATS] [NUM_EXPERIMENTS] [TYPE]
```

- `[NUM_REPEATS]`: number of repetitions per experiment
- `[NUM_EXPERIMENTS]`: number of parallel experiments
- `[TYPE]`: algorithm type (`RTSS`, `IC`, or `LET`)
  - `RTSS`: Passive and active analysis when jitter is a percentage of the period.
  - `IC`: Implicit communication analysis and comparison with $D^{FF}_{base}$ when jitter comes from scheduling. (both passive and active analysis)
  - `LET`: LET communication analysis and comparison with $D^{FF}_{base}$ when jitter comes from scheduling. (both passive and active analysis)

The results are output and saved in active/, passive/, and compare/. The results of different analyses are distinguished by RTSS/IC/LET tags.

The usage examples and running time of run_experiments.sh are as follows:
As a reference, we utilize a machine running Ubuntu 24.04.2 LTS (2025-09-05) x86_64 GNU/Linux, with 13th Gen Intel® Core™ i7-13700F × 24 and 32.0 GiB RAM.

```
# Run 10 LET evaluations in parallel, each with 1000 repeats
bash run_experiments.sh 1000 10 LET  3173.24s user 5.34s system 710% cpu 7:27.49 total
# Run 10 IC evaluations in parallel, each with 1000 repeats
# fig10, fig11, fig13 in the paper can be obtained
bash run_experiments.sh 1000 10 IC  12327.42s user 5.50s system 865% cpu 23:44.21 total
# Run 10 RTSS evaluations in parallel, each with 10 repeats
# fig12, fig14 in the paper can be obtained
bash run_experiments.sh 10 10 RTSS  76818.89s user 4.09s system 633% cpu 3:22:00.65 total
```

Keeping `INITIAL_SEED=1755016037` in `run_experiments.sh` to obtain the same plots from the paper. You can get different results by changing the seed.

### Acknowledgments
This work is partially supported by the project "Trustworthy
Cyber-Physical Pipelines", funded by the MAECI Italy-Sweden
co-operation id. PGR02086, and by VR grant number 2023-06836.
This work is also partially supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation, via the NEST projects Cloud-Robotics: Intelligent Cloud Robotics for Real-Time Manipulation at Scale and by the National Natural Science Foundation of China "Analysis, Modeling and Algorithm Research on Real-time Scheduling Problems of Hybrid Business Flows in Industrial Heterogeneous Networks Based on TSN" (No.62072085), and the China Scholarship Council.

### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
