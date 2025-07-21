# 文件名：main.py
import csv
import datetime
from analysisLET import run_analysis_LET
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from plot import plot_false_percentage_rw
from plot import plot_histogram_rw_from_csv
from analysisLET import RandomEvent_LET
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

    print(f"selected_periods: {selected_periods}, selected_write_offsets: {selected_write_offsets}")
    return selected_periods, selected_read_offsets, selected_write_offsets


def output_results_LET(num_repeats, random_seed, timestamp, run_results_write, false_results_write, run_results_read, false_results_read, num_chains, jitters):

    folder_name = f"{num_repeats}_{random_seed}_{timestamp}"
    folder_path = os.path.join("LET", folder_name)
    os.makedirs(folder_path, exist_ok=True)

    percent_plot_name = os.path.join(folder_path,  f"percent_{num_repeats}_{random_seed}_{timestamp}.png")
    R_plot_name = os.path.join(folder_path, f"R_{num_repeats}_{random_seed}_{timestamp}.png")
    results_csv = os.path.join(folder_path, f"data_{num_repeats}_{random_seed}_{timestamp}.csv" )
    log_txt = os.path.join(f"log/LET_log_{num_repeats}_{random_seed}_{timestamp}.txt")


    # save results to csv
    with open(results_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["seeds","num_tasks", "per_jitter","max_reaction_time", "final_e2e_max_write",  "R_write", "exceed_write", "false_percentage_write", "final_e2e_max_read", "R_read", "exceed_read", "false_percentage_read"])
        for num_tasks in num_chains:
            for per_jitter in jitters:
                false_percentage_write = false_results_write[num_tasks][per_jitter]
                false_percentage_read = false_results_read[num_tasks][per_jitter]
                for (final_e2e_max_read, max_reaction_time, r_read, tasks, seed, exceed_read, final_e2e_max_write, r_write, exceed_write) in zip(run_results_read[num_tasks][per_jitter], run_results_write[num_tasks][per_jitter]):
                    writer.writerow([seed, num_tasks, per_jitter, max_reaction_time, final_e2e_max_write, r_write, exceed_write, false_percentage_write, final_e2e_max_read, r_read, exceed_read, false_percentage_read])

    print(f"All results saved to {results_csv}")

    # save log file
    with open(log_txt, mode='w') as file:
        writer = file.write
        for num_tasks in num_chains:
            for per_jitter in jitters:
                false_percentage_write = false_results_write[num_tasks][per_jitter]
                false_percentage_read = false_results_read[num_tasks][per_jitter]
                writer(f"=====================num_tasks: {num_tasks}, per_jitter: {per_jitter}, false_percentage_write: {false_percentage_write}, false_percentage_read: {false_percentage_read} =====================\n")
                for (final_e2e_max_read, max_reaction_time, r_read, tasks, seed, exceed_read, final_e2e_max_write, r_write, exceed_write) in zip(run_results_read[num_tasks][per_jitter], run_results_write[num_tasks][per_jitter]):
                    writer(f"seed: {seed}, max_reaction_time: {max_reaction_time}, final_e2e_max_write: {final_e2e_max_write}, R_write: {r_write}, {exceed_write}, final_e2e_max_read: {final_e2e_max_read},  R_read: {r_read}, {exceed_read}\n")
                    for task in tasks:
                        writer(f"   {task}\n")
    
    print(f"All results saved to {log_txt}")

    # plotting: uncomment to have plots made automatically
    plot_histogram_rw_from_csv(results_csv, R_plot_name)
    print(f"Plots generated and saved to {R_plot_name}")
    plot_false_percentage_rw(results_csv, percent_plot_name)
    print(f"Plots generated and saved to {percent_plot_name}")

    return results_csv, log_txt, percent_plot_name, R_plot_name



def run_LET(jitters, num_chains, num_repeats, random_seed, periods):
    # preparing list for storing result
    results_write = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    false_results_write = {num_tasks: {per_jitter: 0 for per_jitter in jitters} for num_tasks in num_chains}
    results_read = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    false_results_read = {num_tasks: {per_jitter: 0 for per_jitter in jitters} for num_tasks in num_chains}

    # TODO: add random_seed to the filename
    # run analysis
    for i in range(num_repeats):            # loop on number of repetitions
        random.seed(random_seed)
        for num_tasks in num_chains:        # on number of tasks in a chain
            selected_periods, selected_read_offsets, selected_write_offsets = generate_periods_and_offsets(num_tasks, periods)
            for per_jitter in jitters:      # on relative (to period) magnitude of jitter
                jit = per_jitter * np.array(selected_periods)  # calculate the absolute jitter
                # generate the jitter
                # only generate the jitter
                print(f"================== num_tasks {num_tasks} per_jitter {per_jitter} ({jit}) Repeat {i} random_seed {random_seed} ==================")
                final_e2e_max_write, final_e2e_max_read, max_reaction_time, tasks = run_analysis_LET(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
                # value of rate "= max_reaction_time / final_e2e_max"
                if final_e2e_max_write != 0:
                    r_write = max_reaction_time / final_e2e_max_write
                    if r_write > 1:
                        exceed_write = "exceed"
                    else:
                        exceed_write = "safe"
                else:
                    r_write = None
                    exceed_write = None
                    false_results_write[num_tasks][per_jitter] += 1  # algorithm failed
                if final_e2e_max_read != 0:
                    r_read = max_reaction_time / final_e2e_max_read
                    if r_read > 1:
                        exceed_read = "exceed"
                    else:
                        exceed_read = "safe"
                else:
                    r_read = None
                    exceed_read = None
                    false_results_read[num_tasks][per_jitter] += 1

                results_write[num_tasks][per_jitter].append((final_e2e_max_write, max_reaction_time,r_write,tasks,random_seed,exceed_write))
                results_read[num_tasks][per_jitter].append((final_e2e_max_read, max_reaction_time,r_read,tasks,random_seed,exceed_read))

        random_seed = random_seed+1

    # algorithm2 failed percentage 
    for num_tasks in num_chains:
        for per_jitter in jitters:
            false_percentage_write = (false_results_write[num_tasks][per_jitter] / num_repeats)
            false_results_write[num_tasks][per_jitter] = false_percentage_write
            false_percentage_read = (false_results_read[num_tasks][per_jitter] / num_repeats)
            false_results_read[num_tasks][per_jitter] = false_percentage_read

    return results_write, false_results_write, results_read, false_results_read


    

if __name__ == "__main__":
    # INCREASE here to have more experiments per same settings
    num_repeats = 10  # number of repetitions: if 10 takes about 20 minutes on Shumo's laptop
    # Enrico's laptop: num_repeats=10 ==> 32 seconds
    
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000]  # periods
    
    # jitters = [0,0.01,0.02,0.05,0.1,0.2,0.5,1]  # maxjitter = percent jitter * period
    jitters = [0.02,0.05,0.1,0.2,0.3,0.4,0.5]  # maxjitter = percent jitter * period
    
    # num_chains = [3,5,8,10] 
    num_chains  = [3,5]  # for test
    

    # random_seed = 100  # fixed seed
    # timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime("%Y%m%d_%H%M%S")

    random_seed = int(time.time())
    timestamp = datetime.datetime.fromtimestamp(random_seed).strftime("%Y%m%d_%H%M%S")


    run_results_write, false_results_write, run_results_read, false_results_read = run_LET(jitters, num_chains, num_repeats, random_seed, periods)
    
    output_results_LET(num_repeats, random_seed, timestamp, run_results_write, false_results_write, run_results_read, false_results_read, num_chains, jitters)
    