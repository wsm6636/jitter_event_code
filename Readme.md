# Jitter in Task Chains

The repository is used to reproduce the evaluation from

*"Jitter in Task Chains". Shumo Wang, Enrico Bini, Martina Maggio, Qingxu Deng IEEE Real-Time Systems Symposium (RTSS), 2025*

The tool supports both **passive** and **active** jitter analysis for cause-effect chains, and includes comparison with the state-of-the-art LET and Implicit Communication (IC) models from reference [20].

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
├── active                       #  Active experiment results
├── compare                   # Compare experiment results
├── passive               # Passive experiment results
├── utilities                    # Coppied from [20]
│   ├── analyzer.py              # Methods to analyze end-to-end timing behavior
│   ├── analyzer_our.py              # Methods to analyze mrt, let
│   ├── analyzer_becker.py              # Methods to analyze becker
│   ├── augmented_job_chain.py   # Augmented job chains as in the paper
│   ├── chain.py                 # Cause-effect chains
│   ├── communication.py         # Communication tasks
│   ├── evaluation.py            # Methods to draw plots
│   ├── event_simulator.py       # Event-driven simulator with fixed execution time
│   ├── generator_UUNIFAST       # Task set generator for uunifast benchmark
│   ├── generator_WATERS         # Task set and cause-effect chain generator for waters benchmark
│   ├── task.py                  # Tasks
│   └── transformer.py           # Connect task creating with the scheduler
├── analysis_active.py                     # Active analysis implementation
├── analysis_Gunzel.py                      #  Günzel LET & IC wrappers
├── analysis_passive.py        # Passive analysis implementation
├── evaluation_active.py              #  Active experiment runner
├── evaluation_passive.py        #   Passive experiment runner
├── generate_comparison.py             # Plot & CSV post-processing
├── main.py                      # Main function
├── plot.py                      # Visualization utilities
├── README.md
└── run_experiments.sh  # Batch experiment script
```

The experiments in the main function are splitted into 3 parts:

1. our paper analysis (passive and active)
2. compare our and Gunzel implicit communication experiment (with runtime)
3. compare our and Gunzel LET communication experiment (with runtime)
4. Plotting and spliting the results

The stage result csv files are stored in folders "active/" and "passive". The final merged files, split files, and output images are stored in the folder "compare", and are distinguished by different marks (eg. IC/LET).

### Deployment

The following steps explain how to deploy this framework on the machine:

First, clone the git repository or download
the [zip file](https://github.com/wsm6636/jitter_event_code/archive/refs/heads/additional_experiments.zip):

```
git clone -b additional_experiments https://github.com/wsm6636/jitter_event_code.git
```

### Quick Start

Move into the code folder and execute run_experiments.sh natively:

```
cd jitter_event_code
bash run_experiments.sh [NUM_REPEATS] [NUM_EXPERMENTS] [TYPE]
```

- `[NUM_REPEATS]`: number of repetitions per experiment
- `[NUM_EXPERMENTS]`: number of parallel experiments
- `[TYPE]`: algorithm type (`RTSS`, `IC`, or `LET`)
  - `RTSS`: compare our passive and active analysis
  - `IC`: compare our and Gunzel in implicit communication (both passive and active analysis)
  - `LET`: compare our and Gunzel in LET communication (both passive and active analysis)

Results are stored under `compare/{timestamp}_[TYPE]/`.

As a reference, we utilize a machine running Ubuntu 24.04.2 LTS (2025-09-05) x86_64 GNU/Linux, with 13th Gen Intel® Core™ i7-13700F × 24 and 32.0 GiB RAM.

```
bash run_experiments.sh 1000 10 LET  3173.24s user 5.34s system 710% cpu 7:27.49 total
bash run_experiments.sh 1000 10 IC  12327.42s user 5.50s system 865% cpu 23:44.21 total
bash run_experiments.sh 10 10 RTSS  76818.89s user 4.09s system 633% cpu 3:22:00.65 total
```

Keeping `INITIAL_SEED=1755016037` in `run_experiments.sh` to obtain the same plots from the paper. You can get different results by changing the seed.

### Acknowledgments

### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
