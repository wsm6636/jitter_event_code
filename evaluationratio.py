# 文件名：main.py
import csv
import datetime
from analysisratio import run_analysis_ratio
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from plot import ratio_histogram_from_csv
from plot import ratio_line_chart_from_csv
from plot import ratio_for_num_chains
from analysisratio import RandomEvent_ratio
import random
import time
import os

def generate_periods_and_offsets_ratio(selected_periods):

    # selected_read_offsets = [random.randint(0, (period - 1)) for period in selected_periods]
    selected_read_offsets = [random.uniform(0, period - 1) for period in selected_periods]    
    selected_write_offsets = [read_offset + period for read_offset, period in zip(selected_read_offsets, selected_periods)]

    return selected_read_offsets, selected_write_offsets


def generate_periods(ratio, num_tasks, min_period, max_period):
        
    max_initial_period = (max_period / (ratio ** (num_tasks - 1))) ** (1 / num_tasks)
    initial_period = random.uniform(min_period, min(max_initial_period, max_period))
    print(f"Initial period: {initial_period}, ratio: {ratio}, num_tasks: {num_tasks}, max_initial_period: {max_initial_period}")

    periods = [initial_period]
            
    for n in range(1, num_tasks):
        min_period_n = periods[-1] * ratio
        max_period_n = max_period  / (ratio ** (num_tasks - n - 1))

        new_period = random.uniform(min_period_n, max_period_n)
        print(f"New period for task {n}: {new_period}, min_period_n: {min_period_n}, max_period_n: {max_period_n}")
        periods.append(new_period)

                
    return periods


def output_results_ratio(num_repeats, random_seed, timestamp, results, false_results, num_chains, jitters, ratios):

    folder_name = f"{num_repeats}_{random_seed}_{timestamp}"
    folder_path = os.path.join("ratio", folder_name)
    os.makedirs(folder_path, exist_ok=True)

    percent_plot_name = os.path.join(folder_path, f"percent_{num_repeats}_{random_seed}_{timestamp}.png")
    R_plot_name = os.path.join(folder_path, f"R_{num_repeats}_{random_seed}_{timestamp}.png")
    results_csv = os.path.join(folder_path, f"data_{num_repeats}_{random_seed}_{timestamp}.csv") 
    log_txt = os.path.join(folder_path, f"log_{num_repeats}_{random_seed}_{timestamp}.txt")
    ratio_plot_name = os.path.join(folder_path, f"ratios_{num_repeats}_{timestamp}.png")


    # save results to csv
    with open(results_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["seeds","num_tasks", "per_jitter", "final_e2e_max", "max_reaction_time", "R","exceed", "false_percentage","ratios"])
        for ratio in ratios:
            for num_tasks in num_chains:
                for per_jitter in jitters:
                    false_percentage = false_results[num_tasks][per_jitter][ratio]
                    for (final_e2e_max, max_reaction_time, r, tasks, seed, exceed) in results[num_tasks][per_jitter][ratio]:
                        writer.writerow([seed, num_tasks, per_jitter, final_e2e_max, max_reaction_time, r, exceed, false_percentage, ratio])

    print(f"All results saved to {results_csv}")


    # save log file
    with open(log_txt, mode='w') as file:
        writer = file.write
        for ratio in ratios:
            for num_tasks in num_chains:
                for per_jitter in jitters:
                    false_percentage = false_results[num_tasks][per_jitter][ratio]
                    writer(f"=====================num_tasks: {num_tasks}, per_jitter: {per_jitter}, ratio: {ratio}, false_percentage: {false_percentage}=====================\n")
                    for (final_e2e_max, max_reaction_time, r, tasks, seed, exceed) in results[num_tasks][per_jitter][ratio]:
                        writer(f"seed: {seed}, final_e2e_max: {final_e2e_max}, max_reaction_time: {max_reaction_time}, R: {r}, {exceed}\n")
                        if tasks is not None:
                            for task in tasks:
                                writer(f"   {task}\n")

                        
    print(f"All logs saved to {log_txt}")

    # plotting: uncomment to have plots made automatically
    ratio_histogram_from_csv(results_csv, R_plot_name)
    print(f"Plots generated and saved to {R_plot_name}")
    ratio_line_chart_from_csv(results_csv, percent_plot_name)
    print(f"Plots generated and saved to {percent_plot_name}")
    ratio_for_num_chains(results_csv, ratio_plot_name)
    print(f"Plots generated and saved to {ratio_plot_name}")


def run_ratio(jitters, num_chains, num_repeats, random_seed, ratios, min_period, max_period):
    # preparing list for storing result
    results = {num_tasks: {per_jitter: {ratio: [] for ratio in ratios} for per_jitter in jitters} for num_tasks in num_chains}
    final = {num_tasks: {per_jitter: {ratio: [] for ratio in ratios} for per_jitter in jitters} for num_tasks in num_chains}
    false_results = {num_tasks: {per_jitter: {ratio: 0 for ratio in ratios} for per_jitter in jitters} for num_tasks in num_chains}

    # TODO: add random_seed to the filename
    # run analysis
    for ratio in ratios:
        current_random_seed = random_seed
        for i in range(num_repeats):            # loop on number of repetitions
            random.seed(current_random_seed)        
            for num_tasks in num_chains:        # on number of tasks in a chain
                selected_periods = generate_periods(ratio, num_tasks, min_period, max_period)

                if selected_periods is None:
                    for per_jitter in jitters:
                        print(f"========For ratio {ratio}========== num_tasks {num_tasks} Repeat {i} random_seed {current_random_seed} ==================")
                        results[num_tasks][per_jitter][ratio].append((None, None, None, None, current_random_seed, None))
                        final[num_tasks][per_jitter][ratio].append((None, None))
                        false_results[num_tasks][per_jitter][ratio] += 1
                    continue
                else:
                    selected_read_offsets, selected_write_offsets = generate_periods_and_offsets_ratio(selected_periods) 

                    print(f"========For ratio {ratio}========== selected_period {selected_periods}, ratio {ratio} ==================")
                    for per_jitter in jitters:      # on relative (to period) magnitude of jitter
                        # generate the jitter
                        # only generate the jitter
                        print(f"========For ratio {ratio}========== num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {current_random_seed} ==================")
                        final_e2e_max, max_reaction_time,  final_r, final_w, tasks = run_analysis_ratio(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
                        # value of rate "= max_reaction_time / final_e2e_max"
                        if final_e2e_max != 0:
                            r = max_reaction_time / final_e2e_max
                            if r > 1:
                                exceed = "exceed"
                            else:
                                exceed = "safe"
                        else:
                            r = None
                            exceed = None
                            false_results[num_tasks][per_jitter][ratio] += 1  # algorithm failed

                        results[num_tasks][per_jitter][ratio].append((final_e2e_max, max_reaction_time,r,tasks,current_random_seed,exceed))
                        final[num_tasks][per_jitter][ratio].append((final_r, final_w))
                        
            current_random_seed = current_random_seed+1

    # algorithm2 failed percentage 
    for ratio in ratios:
        for num_tasks in num_chains:
            for per_jitter in jitters:
                false_percentage = (false_results[num_tasks][per_jitter][ratio] / num_repeats)
                false_results[num_tasks][per_jitter][ratio] = false_percentage

    return results, false_results, final

    

if __name__ == "__main__":
    # INCREASE here to have more experiments per same settings
    num_repeats = 1000 # number of repetitions: if 10 takes about 20 minutes on Shumo's laptop
    # Enrico's laptop: num_repeats=10 ==> 32 seconds

    
    # jitters = [0,0.01,0.02,0.05,0.1,0.2,0.5,1]  # maxjitter = percent jitter * period
    jitters = [0,0.02,0.05,0.1,0.2,0.3,0.4,0.5]  # maxjitter = percent jitter * period

    num_chains = [3,5,8,10] 
    # num_chains  = [3,5]  # for test

    min_period = 1  # minimum period
    max_period = 1000  # maximum period

    # ratios = np.arange(1.0, 2.0, 0.5)
    ratios = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    # ratios = [1.0, 1.5, 2.0]
    print(f"Ratios: {ratios}")


    # random_seed = 100  # fixed seed
    # timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime("%Y%m%d_%H%M%S")

    random_seed = int(time.time())
    timestamp = datetime.datetime.fromtimestamp(random_seed).strftime("%Y%m%d_%H%M%S")

    run_ratio_results, false_results, final_task = run_ratio(jitters, num_chains, num_repeats, random_seed, ratios, min_period, max_period)
    output_results_ratio(num_repeats, random_seed, timestamp, run_ratio_results, false_results, num_chains, jitters, ratios)

    