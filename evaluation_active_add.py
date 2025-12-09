#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 05 10:25:52 2025

It implements the methods described in the paper
    "Jitter Propagation in Task Chains". 
    Shumo Wang, Enrico Bini, Qingxu Deng, Martina Maggio, 
    IEEE Real-Time Systems Symposium (RTSS), 2025

@author: Shumo Wang
"""
import csv
import datetime
from analysis_active_add import run_analysis_active_our_add
import random
import time
import os
from evaluation_passive import generate_periods_and_offsets

from plot import plot_R_histogram_our
from plot import plot_false_percent



def output_active_our_add(num_repeats, random_seed, timestamp, results, false_results, num_chains, jitters):
    """
    Write the results of our active experiments (IC/LET=jitter=0) to CSV.
    arguments:
        num_repeats: number of repetitions
        random_seed: the initial random seed
        timestamp: the timestamp of the experiment
        results: the results of the experiments
        false_results: the false results of the experiments
        num_chains: the number of tasks in the chain
        jitters: the list of jitters used in the experiments
    return:
        results_csv: the path to the results CSV file
    """
    folder_path = "add"
    os.makedirs(folder_path, exist_ok=True)

    results_csv = os.path.join(folder_path, f"data_active_add_{num_repeats}_{random_seed}_{timestamp}.csv" )

    # save results to csv
    with open(results_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["seeds","num_tasks", "per_jitter", "final_e2e_max", "max_reaction_time", "R", "exceed", "false_percentage", "added"])
        for num_tasks in num_chains:
            for per_jitter in jitters:
                false_percentage = false_results[num_tasks][per_jitter]
                for (final_e2e_max, max_reaction_time, r, tasks, seed, exceed,added) in results[num_tasks][per_jitter]:
                    writer.writerow([seed,num_tasks, per_jitter, final_e2e_max, max_reaction_time, r, exceed, false_percentage,added])

    print(f"All results saved to {results_csv}")

    return results_csv


"""For testing"""
def run_evaluation_active_our_add(jitters, num_chains, num_repeats, random_seed, periods):
    TOLERANCE = 1e-9
    # preparing list for storing result
    results = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    final = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    false_results = {num_tasks: {per_jitter: 0 for per_jitter in jitters} for num_tasks in num_chains}

    for i in range(num_repeats):            # loop on number of repetitions
        random.seed(random_seed)
        for num_tasks in num_chains:        # on number of tasks in a chain
            selected_periods, selected_read_offsets, selected_write_offsets = generate_periods_and_offsets(num_tasks, periods)
            for per_jitter in jitters:      # on relative (to period) magnitude of jitter
                # generate the jitter
                # only generate the jitter
                print(f"================== num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                final_e2e_max, max_reaction_time,  final_r, final_w, tasks, added = run_analysis_active_our_add(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
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

                results[num_tasks][per_jitter].append((final_e2e_max, max_reaction_time,r,tasks,random_seed,exceed,added))
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
    num_repeats = 10  
    
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000]  # periods
    
    jitters = [0,0.02,0.05,0.1,0.2,0.3,0.4,0.5]  # maxjitter = percent jitter * period
    # jitters = random.choices([0.1,0.2,0.3,0.4,0.5], k=1)
    # jitters = [0.4]  # for test
    num_chains = [3,5,8,10] 
    # num_chains  = [3]  # for test

    random_seed = 1755016037
    # random_seed =  int(time.time())
    timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime("%Y%m%d_%H%M%S")

    # random_seed = int(time.time())
    # timestamp = datetime.datetime.fromtimestamp(random_seed).strftime("%Y%m%d_%H%M%S")

    results, false_results, final = run_evaluation_active_our_add(jitters, num_chains, num_repeats, random_seed, periods)
    output_active_our_add(num_repeats, random_seed, timestamp, results, false_results, num_chains, jitters)


