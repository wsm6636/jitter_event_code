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
from analysis_passive_order import run_analysis_passive_our_order
import random
import time
import os
from plot import plot_false_percent_order
from plot import plot_R_histogram_our_order
from plot import plot_type_percent_order    
import numpy as np

def generate_LET(num_tasks, periods):
    """
    Generates LET task parameters:
    The read time for each task is fixed at the beginning of the period, and the write time is fixed at the end of the period.
    arguments:
        num_tasks: number of tasks in the chain
        periods: list of possible periods
    return:
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
    arguments:
        num_tasks: number of tasks in the chain
        periods: list of possible periods
    return:
        selected_periods: A list of periods of length = num_tasks
        selected_read_offsets: A list of random read offsets
        selected_write_offsets: Each element = read_offset + period
    """  
    selected_periods = random.choices(periods,  k=num_tasks)
    selected_read_offsets = [random.randint(0, period) for period in selected_periods]
    selected_write_offsets = [read_offset + period for read_offset, period in zip(selected_read_offsets, selected_periods)]

    print(f"selected_periods: {selected_periods}, selected_read_offsets: {selected_read_offsets}, selected_write_offsets: {selected_write_offsets}")
    return selected_periods, selected_read_offsets, selected_write_offsets





def output_passive_our_order(num_repeats, random_seed, timestamp, results, false_results, num_chains, jitters, chain_types, false_percentage_by_chain_type, false_percentage_by_num_tasks):
    """
    Write the results of our passive experiments (IC/LET=jitter=0) to CSV.
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
    folder_path = "order/passive"
    os.makedirs(folder_path, exist_ok=True)

    results_csv = os.path.join(folder_path, f"data_passive_order_{num_repeats}_{random_seed}_{timestamp}.csv" )

    percent_plot_name = os.path.join(folder_path,  f"percent_passive_order_{num_repeats}_{random_seed}_{timestamp}.png")
    R_plot_name = os.path.join(folder_path, f"R_passive_order_{num_repeats}_{random_seed}_{timestamp}.png")
    type_plot_name = os.path.join(folder_path, f"type_passive_order_{num_repeats}_{random_seed}_{timestamp}.png")

    # save results to csv
    with open(results_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["seeds","num_tasks", "per_jitter", "chain_type", "final_e2e_max", "max_reaction_time", "R", "exceed", "false_percentage_global",
            "false_percentage_by_chain_type",
            "false_percentage_by_num_tasks"])
        for result in results:
            num_tasks_index = num_chains.index(result['num_tasks'])
            per_jitter_index = jitters.index(result['per_jitter'])
            chain_index = chain_types.index(result['chain_type'])
            # false_percentage = false_results[num_tasks_index, per_jitter_index, chain_index]

            # 全局失败率（原来的）
            false_percentage_global = false_results[num_tasks_index, per_jitter_index, chain_index]
            
            # 按chain_type分组的失败率
            false_percentage_chain_type = false_percentage_by_chain_type[result['chain_type']][result['num_tasks']][result['per_jitter']]
            
            # 按num_tasks分组的失败率  
            false_percentage_num_tasks = false_percentage_by_num_tasks[result['num_tasks']][result['chain_type']][result['per_jitter']]

            writer.writerow([
                result['seed'],
                result['num_tasks'],
                result['per_jitter'],
                result['chain_type'],
                result['final_e2e_max'],
                result['max_reaction_time'],
                result['R'],
                result['exceed'],
                false_percentage_global,
                false_percentage_chain_type,
                false_percentage_num_tasks
            ])

    print(f"All results saved to {results_csv}")

    # plot false percentage
    plot_false_percent_order(percent_plot_name, results_csv)
    plot_R_histogram_our_order(R_plot_name, results_csv)
    plot_type_percent_order(type_plot_name, results_csv)
    return results_csv



def run_evaluation_passive_our_order(jitters, num_chains, num_repeats, random_seed, periods, chain_types):
    # Preparing lists for storing results
    results = []
    TOLERANCE = 1e-9
    # Initialize false_results array with zeros
    false_results = np.zeros((len(num_chains), len(jitters), len(chain_types)))


    # Run analysis
    for i in range(num_repeats):  # Loop on number of repetitions
        random.seed(random_seed)

        for num_tasks_index, num_tasks in enumerate(num_chains):  # Loop on number of tasks in a chain
            selected_periods, selected_read_offsets, selected_write_offsets = generate_periods_and_offsets(num_tasks, periods)

            for per_jitter_index, per_jitter in enumerate(jitters):  # Loop on relative (to period) magnitude of jitter
                print(f"================== num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                chain_results, max_reaction_time, tasks = run_analysis_passive_our_order(num_tasks, selected_periods, selected_read_offsets, selected_write_offsets, per_jitter, chain_types)

                for chain_index, chain_type in enumerate(chain_types):
                    final_e2e_max, _, _ = chain_results[chain_type]
                    if final_e2e_max != 0:
                        r = max_reaction_time / final_e2e_max
                        if r > 1 + TOLERANCE:  # if rate is larger than 1, then algorithm failed
                            exceed = "exceed"
                        else:
                            exceed = "safe"
                    else:
                        r = None
                        exceed = None
                        false_results[num_tasks_index, per_jitter_index, chain_index] += 1  # Algorithm failed

                    results.append({
                        'seed': random_seed,
                        'num_tasks': num_tasks,
                        'per_jitter': per_jitter,
                        'chain_type': chain_type,
                        'final_e2e_max': final_e2e_max,
                        'max_reaction_time': max_reaction_time,
                        'R': r,
                        'exceed': exceed,
                        'tasks': tasks
                    })

        random_seed += 1

    # Calculate false percentage
    false_results = false_results / num_repeats

    # 计算按chain_type分组的失败率字典
    false_percentage_by_chain_type = {}
    for chain_index, chain_type in enumerate(chain_types):
        false_percentage_by_chain_type[chain_type] = {}
        for num_tasks_index, num_tasks in enumerate(num_chains):
            false_percentage_by_chain_type[chain_type][num_tasks] = {}
            for per_jitter_index, per_jitter in enumerate(jitters):
                false_percentage_by_chain_type[chain_type][num_tasks][per_jitter] = false_results[num_tasks_index, per_jitter_index, chain_index]

    # 计算按num_tasks分组的失败率字典
    false_percentage_by_num_tasks = {}
    for num_tasks_index, num_tasks in enumerate(num_chains):
        false_percentage_by_num_tasks[num_tasks] = {}
        for chain_index, chain_type in enumerate(chain_types):
            false_percentage_by_num_tasks[num_tasks][chain_type] = {}
            for per_jitter_index, per_jitter in enumerate(jitters):
                false_percentage_by_num_tasks[num_tasks][chain_type][per_jitter] = false_results[num_tasks_index, per_jitter_index, chain_index]

    return results, false_results, false_percentage_by_chain_type, false_percentage_by_num_tasks





    

if __name__ == "__main__":
    num_repeats = 10  
    
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000]  # periods
    
    jitters = [0,0.02,0.05,0.1,0.2,0.3,0.4,0.5]  # maxjitter = percent jitter * period
    # jitters = [0,0.02,0.05,0.1,0.2]

    num_chains = [3,5,8,10] 
    # num_chains = [3,5]
    
    random_seed = 1755016037  # fixed seed
    timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime("%Y%m%d_%H%M%S")

    # random_seed = int(time.time())
    # timestamp = datetime.datetime.fromtimestamp(random_seed).strftime("%Y%m%d_%H%M%S")

    chain_types = ['asc', 'desc', 'max_period', 'min_period']

    run_results, false_results, false_percentage_by_chain_type, false_percentage_by_num_tasks = run_evaluation_passive_our_order(jitters, num_chains, num_repeats, random_seed, periods, chain_types)

    output_passive_our_order(num_repeats, random_seed, timestamp, run_results, false_results, num_chains, jitters, chain_types, false_percentage_by_chain_type, false_percentage_by_num_tasks)


