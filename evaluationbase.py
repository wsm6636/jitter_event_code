# 文件名：main.py
import csv
import datetime
from analysisbase import run_analysis_base
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from plot import plot_baseline_from_csv
from analysisbase import RandomEvent_base
import random
import time
import os

def generate_periods_and_offsets(num_tasks, periods):
    """
    generate periods and offsets for tasks
    :param num_tasks: number of tasks
    :param periods: list of periods
    :param jitter_factor: jitter percentage
    :param seed: random seed
    :return: periods, read_offsets, write_offsets
    """  
    selected_periods = random.choices(periods,  k=num_tasks)
    # selected_read_offsets = [random.randint(0, (period - 1)) for period in selected_periods]
    selected_read_offsets = [random.uniform(0, period) for period in selected_periods]  
    selected_write_offsets = [read_offset + period for read_offset, period in zip(selected_read_offsets, selected_periods)]

    print(f"selected_periods: {selected_periods}, selected_read_offsets: {selected_read_offsets}, selected_write_offsets: {selected_write_offsets}")
    return selected_periods, selected_read_offsets, selected_write_offsets


def output_results(num_repeats, random_seed, timestamp, results, num_chains, jitters):

    folder_name = f"{num_repeats}_{random_seed}_{timestamp}"
    folder_path = os.path.join("base", folder_name)
    os.makedirs(folder_path, exist_ok=True)

    percent_plot_name = os.path.join(folder_path,  f"baseline_{num_repeats}_{random_seed}_{timestamp}.png")
    results_csv = os.path.join(folder_path, f"data_{num_repeats}_{random_seed}_{timestamp}.csv" )
    log_txt = os.path.join(f"log/base_log_{num_repeats}_{random_seed}_{timestamp}.txt")


    # save results to csv
    with open(results_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["seeds","num_tasks", "per_jitter", "max_reaction_time"])
        for num_tasks in num_chains:
            for per_jitter in jitters:
                for (max_reaction_time, tasks, seed) in results[num_tasks][per_jitter]:
                    writer.writerow([seed,num_tasks, per_jitter, max_reaction_time])

    print(f"All results saved to {results_csv}")

    # save log file
    with open(log_txt, mode='w') as file:
        writer = file.write
        for num_tasks in num_chains:
            for per_jitter in jitters:
                writer(f"=====================num_tasks: {num_tasks}, per_jitter: {per_jitter}=====================\n")
                for (max_reaction_time, tasks, seed) in results[num_tasks][per_jitter]:
                    writer(f"seed: {seed}, max_reaction_time: {max_reaction_time}\n")
                    for task in tasks:
                        writer(f"   {task}\n")
    
    print(f"All results saved to {log_txt}")

    plot_baseline_from_csv(results_csv, percent_plot_name)
    print(f"Plots generated and saved to {percent_plot_name}")

    return results_csv, log_txt, percent_plot_name



def run(jitters, num_chains, num_repeats, random_seed, periods):
    # preparing list for storing result
    results = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    config = {}
    for num_tasks in num_chains:
        random.seed(random_seed)
        config[num_tasks] = generate_periods_and_offsets(num_tasks, periods)
    # TODO: add random_seed to the filename
    # run analysis
    index = 0
    for num_tasks in num_chains:
        selected_periods, selected_read_offsets, selected_write_offsets = config[num_tasks]
        for per_jitter in jitters:
            for i in range(num_repeats):  # loop on number of repetitions
                seed_now = random_seed + index
                random.seed(seed_now)
                print(f"================== num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {seed_now} ==================")
                max_reaction_time, tasks = run_analysis_base(num_tasks, selected_periods, selected_read_offsets, selected_write_offsets, per_jitter)
                results[num_tasks][per_jitter].append((max_reaction_time, tasks, seed_now))
                index += 1

    return results


    

if __name__ == "__main__":
    # INCREASE here to have more experiments per same settings
    num_repeats = 200  # number of repetitions: if 10 takes about 20 minutes on Shumo's laptop
    # Enrico's laptop: num_repeats=10 ==> 32 seconds
    
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000]  # periods
    
    # jitters = [0,0.01,0.02,0.05,0.1,0.2,0.5,1]  # maxjitter = percent jitter * period
    # jitters = [0,0.02,0.05,0.1,0.2,0.3,0.4,0.5]  # maxjitter = percent jitter * period
    jitters = [round(x * 0.01, 2) for x in range(51)]  # 0–50 % 步长 1 %
    num_chains = [3,5,8,10] 
    # num_chains  = [3,5]  # for test
    

    # random_seed = 100  # fixed seed
    # timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime("%Y%m%d_%H%M%S")

    random_seed = int(time.time())
    timestamp = datetime.datetime.fromtimestamp(random_seed).strftime("%Y%m%d_%H%M%S")


    run_results = run(jitters, num_chains, num_repeats, random_seed, periods)
    
    output_results(num_repeats, random_seed, timestamp, run_results, num_chains, jitters)
    