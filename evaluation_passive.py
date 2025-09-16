#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 05 10:25:52 2025

It implements the methods described in the paper
    Shumo Wang, Enrico Bini, Martina Maggio, Qingxu Deng
    "Jitter Propagation in Task Chains"

@author: Shumo Wang
"""
import csv
import datetime
from analysis_passive import run_analysis_passive_our
import random
import time
import os


def generate_LET(num_tasks, periods):
    """
    Generates LET task parameters:
    The read time for each task is fixed at the beginning of the period, and the write time is fixed at the end of the period.
    Return value:
        selected_periods: A list of periods of length = num_tasks
        selected_read_offsets: A list of all zeros
        selected_write_offsets: Each element = read_offset + period
    """
    selected_periods = random.choices(periods,  k=num_tasks)
    selected_read_offsets = [0 for period in selected_periods]
    selected_write_offsets = [read_offset + period for read_offset, period in zip(selected_read_offsets, selected_periods)]

    print(f"selected_periods: {selected_periods}, selected_read_offsets: {selected_read_offsets}, selected_write_offsets: {selected_write_offsets}")
    return selected_periods, selected_read_offsets, selected_write_offsets



def generate_periods_and_offsets(num_tasks, periods):
    """
    Generate periods and offsets for tasks
    Return value:
        selected_periods: A list of periods of length = num_tasks
        selected_read_offsets: A list of random read offsets
        selected_write_offsets: Each element = read_offset + period
    """  
    selected_periods = random.choices(periods,  k=num_tasks)
    selected_read_offsets = [random.randint(0, period) for period in selected_periods]
    selected_write_offsets = [read_offset + period for read_offset, period in zip(selected_read_offsets, selected_periods)]

    print(f"selected_periods: {selected_periods}, selected_read_offsets: {selected_read_offsets}, selected_write_offsets: {selected_write_offsets}")
    return selected_periods, selected_read_offsets, selected_write_offsets



def output_passive_Gunzel_IC(num_repeats, random_seed, timestamp, results, false_results, num_chains):
    """
    Write the results of "Gunzel vs our" passive experiments on IC to CSV.
    """
    folder_path = "passive"
    os.makedirs(folder_path, exist_ok=True)

    results_csv = os.path.join(folder_path, f"data_passive_IC_{num_repeats}_{random_seed}_{timestamp}.csv" )
    # save results to csv
    with open(results_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["seeds","num_tasks",  "final_e2e_max", "ic", "R", "exceed", "false_percentage","run_time_our","run_time_G"])
        for num_tasks in num_chains:
            false_percentage = false_results[num_tasks]
            for (final_e2e_max, ic, r, tasks, seed, exceed,run_time_our,run_time_G) in results[num_tasks]:
                writer.writerow([seed,num_tasks, final_e2e_max, ic, r, exceed, false_percentage,run_time_our,run_time_G])

    print(f"All results saved to {results_csv}")
    return results_csv



def output_passive_Gunzel_LET(num_repeats, random_seed, timestamp, results, false_results, num_chains, jitters):
    """
    Write the results of "Gunzel vs our" passive experiments on LET to CSV.
    """
    folder_path = "passive"
    os.makedirs(folder_path, exist_ok=True)

    results_csv = os.path.join(folder_path, f"data_passive_LET_{num_repeats}_{random_seed}_{timestamp}.csv" )

    # save results to csv
    with open(results_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["seeds","num_tasks",  "final_e2e_max", "let", "R", "exceed", "false_percentage","run_time_our","run_time_G"])
        for num_tasks in num_chains:
            for per_jitter in jitters:
                false_percentage = false_results[num_tasks][per_jitter]
                for (final_e2e_max, let, r, tasks, seed, exceed,run_time_our,run_time_G) in results[num_tasks][per_jitter]:
                    writer.writerow([seed,num_tasks, final_e2e_max, let, r, exceed, false_percentage,run_time_our,run_time_G])

    print(f"All results saved to {results_csv}")
    return results_csv



def output_passive_our(num_repeats, random_seed, timestamp, results, false_results, num_chains, jitters):
    """
    Write the results of our passive experiments (IC/LET=jitter=0) to CSV.
    """
    folder_path = "passive"
    os.makedirs(folder_path, exist_ok=True)

    results_csv = os.path.join(folder_path, f"data_passive_{num_repeats}_{random_seed}_{timestamp}.csv" )

    # save results to csv
    with open(results_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["seeds","num_tasks", "per_jitter", "final_e2e_max", "max_reaction_time", "R", "exceed", "false_percentage"])
        for num_tasks in num_chains:
            for per_jitter in jitters:
                false_percentage = false_results[num_tasks][per_jitter]
                for (final_e2e_max, max_reaction_time, r, tasks, seed, exceed) in results[num_tasks][per_jitter]:
                    writer.writerow([seed,num_tasks, per_jitter, final_e2e_max, max_reaction_time, r, exceed, false_percentage])

    print(f"All results saved to {results_csv}")

    return results_csv


"""For testing"""
def run_evaluation_passive_our(jitters, num_chains, num_repeats, random_seed, periods):
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
                final_e2e_max, max_reaction_time,  final_r, final_w, tasks = run_analysis_passive_our(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
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

                results[num_tasks][per_jitter].append((final_e2e_max, max_reaction_time,r,tasks,random_seed,exceed))
                final[num_tasks][per_jitter].append((final_r, final_w))

        random_seed = random_seed+1

    # algorithm2 failed percentage 
    for num_tasks in num_chains:
        for per_jitter in jitters:
            false_percentage = (false_results[num_tasks][per_jitter] / num_repeats)
            false_results[num_tasks][per_jitter] = false_percentage

    return results, false_results, final


    

if __name__ == "__main__":
    num_repeats = 10  
    
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000]  # periods
    
    jitters = [0,0.02,0.05,0.1,0.2,0.3,0.4,0.5]  # maxjitter = percent jitter * period
    
    num_chains = [3,5,8,10] 
    
    random_seed = 1755016037  # fixed seed
    timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime("%Y%m%d_%H%M%S")

    # random_seed = int(time.time())
    # timestamp = datetime.datetime.fromtimestamp(random_seed).strftime("%Y%m%d_%H%M%S")

    run_evaluation_passive_our(jitters, num_chains, num_repeats, random_seed, periods)
    
