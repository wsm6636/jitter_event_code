# 文件名：main.py
import csv
import datetime
from analysis import run_analysis
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from plot import plot_histogram_from_csv 
from plot import plot_line_chart_from_csv
from analysis import RandomEvent
import random
import time


def generate_periods_and_offsets(num_tasks, periods, seed):
    """
    generate periods and offsets for tasks
    :param num_tasks: number of tasks
    :param periods: list of periods
    :param jitter_factor: jitter percentage
    :param seed: random seed
    :return: periods, read_offsets, write_offsets
    """
    random.seed(seed)  
    selected_periods = random.choices(periods,  k=num_tasks)
    selected_read_offsets = [random.randint(0, (period - 1)) for period in selected_periods]  
    # selected_write_offsets = [read_offset + period for read_offset, period in zip(selected_read_offsets, selected_periods)]
    selected_write_offsets = [random.randint(read_offset, read_offset+period) for read_offset, period in zip(selected_read_offsets, selected_periods)]
    # print(f"selected_periods: {selected_periods}, selected_read_offsets: {selected_read_offsets}, selected_write_offsets: {selected_write_offsets}, seed: {seed}")
    return selected_periods, selected_read_offsets, selected_write_offsets



def main():
    # INCREASE here to have more experiments per same settings
    num_repeats = 5  # number of repetitions: if 10 takes about 20 minutes on Shumo's laptop
    # Enrico's laptop: num_repeats=10 ==> 32 seconds
    
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000]  # periods
    
    # jitters = [0,0.01,0.02,0.05,0.1,0.2,0.5,1]  # maxjitter = percent jitter * period
    jitters = [0,0.02,0.05,0.1,0.2,0.3,0.4,0.5]  # maxjitter = percent jitter * period
    
    # num_chains = [3,5,8,10] 
    num_chains  = [3,5]  

    # below we are setting the random seed. Depending on the need, it may be set to a fixed value or a time-dependent value
    # RANDOM SEED: set it to time to avoid repetition. Or to a given value for reproducibility
    random_seed = time.time()
    # random_seed = 100  # fixed seed

    # name for log file
    timestamp = datetime.datetime.fromtimestamp(random_seed).strftime("%Y%m%d_%H%M%S")
    percent_plot_name = f"percent_{num_repeats}_{timestamp}.png"
    R_plot_name = f"R_{num_repeats}_{timestamp}.png"
    results_csv = f"data_{num_repeats}_{timestamp}.csv" 

    # preparing list for storing result
    results = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    final = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    false_results = {num_tasks: {per_jitter: 0 for per_jitter in jitters} for num_tasks in num_chains}

    # TODO: add random_seed to the filename
    # run analysis
    for i in range(num_repeats):            # loop on number of repetitions
        # random.seed(random_seed)
        for num_tasks in num_chains:        # on number of tasks in a chain
            selected_periods, selected_read_offsets, selected_write_offsets = generate_periods_and_offsets(num_tasks, periods, random_seed)
            for per_jitter in jitters:      # on relative (to period) magnitude of jitter
                # generate the jitter
                # only generate the jitter
                print(f"================== num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                final_e2e_max, max_reaction_time,  final_r, final_w, tasks = run_analysis(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
                # value of rate "= max_reaction_time / final_e2e_max"
                if final_e2e_max != 0:
                    r = max_reaction_time / final_e2e_max
                else:
                    r = None
                    false_results[num_tasks][per_jitter] += 1  # algorithm failed

                results[num_tasks][per_jitter].append((final_e2e_max, max_reaction_time,r,tasks))
                final[num_tasks][per_jitter].append((final_r, final_w))
        random_seed = random_seed+1

    # algorithm2 failed percentage 
    for num_tasks in num_chains:
        for per_jitter in jitters:
            false_percentage = (false_results[num_tasks][per_jitter] / num_repeats) * 100
            false_results[num_tasks][per_jitter] = false_percentage

    # save results to csv
    with open(results_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["num_tasks", "per_jitter", "final_e2e_max", "max_reaction_time", "R", "false_percentage"])
        for num_tasks in num_chains:
            for per_jitter in jitters:
                false_percentage = false_results[num_tasks][per_jitter]
                for (final_e2e_max, max_reaction_time, r, _) in results[num_tasks][per_jitter]:
                    writer.writerow([num_tasks, per_jitter, final_e2e_max, max_reaction_time, r, false_percentage])

    print(f"All results saved to {results_csv}")

    # plotting: uncomment to have plots made automatically
    plot_histogram_from_csv(results_csv, R_plot_name)
    print(f"Plots generated and saved to {R_plot_name}")
    plot_line_chart_from_csv(results_csv, percent_plot_name)
    print(f"Plots generated and saved to {percent_plot_name}")

if __name__ == "__main__":
    main()
    