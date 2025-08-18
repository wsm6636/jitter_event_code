# 文件名：main.py
import csv
import datetime
import sys

import pandas as pd
from analysisC1 import run_analysis_C1
from plot import plot_histogram_from_csv 
from plot import plot_line_chart_from_csv
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


def output_results_C1(num_repeats, random_seed, timestamp, results, false_results, num_chains, jitters):

    # folder_name = f"{num_repeats}_{random_seed}_{timestamp}"
    # folder_path = os.path.join("C1", folder_name)
    folder_path = "C1"
    os.makedirs(folder_path, exist_ok=True)

    percent_plot_name = os.path.join(folder_path,  f"percent_{num_repeats}_{random_seed}_{timestamp}.png")
    R_plot_name = os.path.join(folder_path, f"R_{num_repeats}_{random_seed}_{timestamp}.png")
    results_csv = os.path.join(folder_path, f"data_{num_repeats}_{random_seed}_{timestamp}.csv" )
    log_txt = os.path.join(f"log/C1_log_{num_repeats}_{random_seed}_{timestamp}.txt")


    # save results to csv
    with open(results_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["seeds","num_tasks", "per_jitter", "final_e2e_max", "max_reaction_time", "R", "exceed", "false_percentage", "adjust", "inserted"])
        for num_tasks in num_chains:
            for per_jitter in jitters:
                false_percentage = false_results[num_tasks][per_jitter]
                for (final_e2e_max, max_reaction_time, r, tasks, seed, exceed,adjust,inserted) in results[num_tasks][per_jitter]:
                    writer.writerow([seed,num_tasks, per_jitter, final_e2e_max, max_reaction_time, r, exceed, false_percentage,adjust,inserted])

    print(f"All results saved to {results_csv}")

    # save log file
    with open(log_txt, mode='w') as file:
        writer = file.write
        for num_tasks in num_chains:
            for per_jitter in jitters:
                false_percentage = false_results[num_tasks][per_jitter]
                writer(f"=====================num_tasks: {num_tasks}, per_jitter: {per_jitter}, false_percentage: {false_percentage}=====================\n")
                for (final_e2e_max, max_reaction_time, r, tasks, seed, exceed, adjust, inserted) in results[num_tasks][per_jitter]:
                    writer(f"seed: {seed}, final_e2e_max: {final_e2e_max}, max_reaction_time: {max_reaction_time}, R: {r}, {exceed}, adjust: {adjust}, inserted: {inserted}\n")
                    for task in tasks:
                        writer(f"   {task}\n")
    
    print(f"All results saved to {log_txt}")

    # plotting: uncomment to have plots made automatically
    # plot_histogram_from_csv(results_csv, R_plot_name)
    # print(f"Plots generated and saved to {R_plot_name}")
    # plot_line_chart_from_csv(results_csv, percent_plot_name)
    # print(f"Plots generated and saved to {percent_plot_name}")

    return results_csv, log_txt, percent_plot_name, R_plot_name



def filter_and_export_csv_C1(csv_file_path, num_chains, data_output_dir=None):
    if data_output_dir is None:
    # Get parent directory of the CSV file and create data subdirectory
        parent_dir = os.path.dirname(csv_file_path)
        data_dir = os.path.join(parent_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
    
    # Read the original CSV file
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: File {csv_file_path} not found.")
        return [], []
    
    all_data_files = []
    jitter_20_files = []
    
    # Process each num_tasks value
    for num_tasks in num_chains:
        # Filter all data for current num_tasks
        all_data = df[df['num_tasks'] == num_tasks]
        
        # Filter data for current num_tasks with per_jitter = 20%
        jitter_20_data = df[(df['num_tasks'] == num_tasks) & (df['per_jitter'] == 0.2)]
        
        # Define file paths
        all_data_file = os.path.join(data_output_dir, f"C1_data{num_tasks}.csv")
        jitter_20_file = os.path.join(data_output_dir, f"C1_data{num_tasks}_20per.csv")
        
        # Save all data for current num_tasks
        if not all_data.empty:
            all_data.to_csv(all_data_file, index=False)
            all_data_files.append(all_data_file)
            print(f"All data for {num_tasks} tasks saved to {all_data_file}")
        else:
            print(f"Warning: No data found for {num_tasks} tasks")
        
        # Save 20% jitter data for current num_tasks
        if not jitter_20_data.empty:
            jitter_20_data.to_csv(jitter_20_file, index=False)
            jitter_20_files.append(jitter_20_file)
            print(f"20% jitter data for {num_tasks} tasks saved to {jitter_20_file}")
        else:
            print(f"Warning: No 20% jitter data found for {num_tasks} tasks")
    
    return all_data_files, jitter_20_files


def run_C1(jitters, num_chains, num_repeats, random_seed, periods):
    TOLERANCE = 1e-9
    # preparing list for storing result
    results = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    final = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    false_results = {num_tasks: {per_jitter: 0 for per_jitter in jitters} for num_tasks in num_chains}

    # TODO: add random_seed to the filename
    # run analysis
    for i in range(num_repeats):            # loop on number of repetitions
        random.seed(random_seed)
        for num_tasks in num_chains:        # on number of tasks in a chain
            selected_periods, selected_read_offsets, selected_write_offsets = generate_periods_and_offsets(num_tasks, periods)
            for per_jitter in jitters:      # on relative (to period) magnitude of jitter
                # generate the jitter
                # only generate the jitter
                print(f"================== num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                final_e2e_max, max_reaction_time,  final_r, final_w, tasks, adjust, inserted = run_analysis_C1(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
                # value of rate "= max_reaction_time / final_e2e_max"
                if final_e2e_max != 0:
                    r = max_reaction_time / final_e2e_max
                    if r > 1 + TOLERANCE:  # if rate is larger than 1, then algorithm failed
                        exceed = "exceed"
                    else:
                        exceed = "safe"
                else:
                    r = None
                    exceed = None
                    false_results[num_tasks][per_jitter] += 1  # algorithm failed

                results[num_tasks][per_jitter].append((final_e2e_max, max_reaction_time,r,tasks,random_seed,exceed,adjust,inserted))
                final[num_tasks][per_jitter].append((final_r, final_w))

        random_seed = random_seed+1

    # algorithm2 failed percentage 
    for num_tasks in num_chains:
        for per_jitter in jitters:
            false_percentage = (false_results[num_tasks][per_jitter] / num_repeats)
            false_results[num_tasks][per_jitter] = false_percentage

    return results, false_results, final


    

if __name__ == "__main__":
    # INCREASE here to have more experiments per same settings
    # num_repeats = 1  
    
    # periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000]  # periods
    
    # jitters = [0,0.02,0.05,0.1,0.2,0.3,0.4,0.5]  # maxjitter = percent jitter * period

    # num_chains = [3,5,8,10] 
    # num_chains  = [3,5]  # for test

    # random_seed = 1754657734
    # timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime("%Y%m%d_%H%M%S")

    # random_seed = int(time.time())
    # timestamp = datetime.datetime.fromtimestamp(random_seed).strftime("%Y%m%d_%H%M%S")


    # run_results, false_results, final_task = run_C1(jitters, num_chains, num_repeats, random_seed, periods)
    
    # output_results_C1(num_repeats, random_seed, timestamp, run_results, false_results, num_chains, jitters)
    
    num_chains = [3,5,8,10] 
    csv_file_path = sys.argv[1]
    
    filter_and_export_csv_C1(csv_file_path, num_chains)
