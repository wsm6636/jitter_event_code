# 文件名：main.py
import csv
import datetime
from analysisorder import run_analysis
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from plot import plot_percent_order 
from plot import plot_r_histogram_order
from analysisorder import RandomEvent
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


def output_results(num_repeats, random_seed, timestamp, results, false_results, num_chains, jitters):
    folder_name = f"{num_repeats}_{random_seed}_{timestamp}"
    folder_path = os.path.join("order", folder_name)
    os.makedirs(folder_path, exist_ok=True)

    percent_plot_name = os.path.join(folder_path,  f"percent_{num_repeats}_{random_seed}_{timestamp}.png")
    R_plot_name = os.path.join(folder_path, f"R_{num_repeats}_{random_seed}_{timestamp}.png")
    results_csv = os.path.join(folder_path, f"data_{num_repeats}_{random_seed}_{timestamp}.csv" )
    log_txt = os.path.join(f"log/order_log_{num_repeats}_{random_seed}_{timestamp}.txt")

    # Define chain types
    chain_types = ['asc', 'desc', 'max_period', 'min_period']

    # Save results to CSV
    with open(results_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["seed", "num_tasks", "per_jitter", "chain_type", "final_e2e_max", "max_reaction_time", "R", "exceed", "false_percentage"])
        
        for result in results:
            num_tasks_index = num_chains.index(result['num_tasks'])
            per_jitter_index = jitters.index(result['per_jitter'])
            chain_index = chain_types.index(result['chain_type'])
            false_percentage = false_results[num_tasks_index, per_jitter_index, chain_index]
            writer.writerow([
                result['seed'],
                result['num_tasks'],
                result['per_jitter'],
                result['chain_type'],
                result['final_e2e_max'],
                result['max_reaction_time'],
                result['R'],
                result['exceed'],
                false_percentage
            ])

    print(f"All results saved to {results_csv}")

    # Save log file
    with open(log_txt, mode='w') as file:
        for result in results:
            num_tasks_index = num_chains.index(result['num_tasks'])
            per_jitter_index = jitters.index(result['per_jitter'])
            chain_index = chain_types.index(result['chain_type'])
            false_percentage = false_results[num_tasks_index, per_jitter_index, chain_index]
            file.write(f"=====================num_tasks: {result['num_tasks']}, per_jitter: {result['per_jitter']}, chain_type: {result['chain_type']}, false_percentage: {false_percentage}=====================\n")
            file.write(f"seed: {result['seed']}, final_e2e_max: {result['final_e2e_max']}, max_reaction_time: {result['max_reaction_time']}, R: {result['R']}, exceed: {result['exceed']}\n")
            for task in result['tasks']:
                file.write(f"   {task}\n")
    
    print(f"All results saved to {log_txt}")

    plot_percent_order(percent_plot_name, results_csv)
    plot_r_histogram_order(R_plot_name, results_csv)
    print(f"Plots generated and saved to {percent_plot_name} and {R_plot_name}")

    return results_csv, log_txt

def run(jitters, num_chains, num_repeats, random_seed, periods):
    # Preparing lists for storing results
    results = []

    # Initialize false_results array with zeros
    false_results = np.zeros((len(num_chains), len(jitters), 4))  

    # Define chain types
    chain_types = ['asc', 'desc', 'max_period', 'min_period']

    # Run analysis
    for i in range(num_repeats):  # Loop on number of repetitions
        random.seed(random_seed)
        for num_tasks_index, num_tasks in enumerate(num_chains):  # Loop on number of tasks in a chain
            selected_periods, selected_read_offsets, selected_write_offsets = generate_periods_and_offsets(num_tasks, periods)
            for per_jitter_index, per_jitter in enumerate(jitters):  # Loop on relative (to period) magnitude of jitter
                print(f"================== num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                chain_results, max_reaction_time, tasks = run_analysis(num_tasks, selected_periods, selected_read_offsets, selected_write_offsets, per_jitter)

                for chain_index, chain_type in enumerate(chain_types):
                    final_e2e_max, _, _ = chain_results[chain_type]
                    if final_e2e_max != 0:
                        r = max_reaction_time / final_e2e_max
                        exceed = "exceed" if r > 1 else "safe"
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

    return results, false_results


if __name__ == "__main__":
    # INCREASE here to have more experiments per same settings
    num_repeats = 2  # number of repetitions: if 10 takes about 20 minutes on Shumo's laptop
    # Enrico's laptop: num_repeats=10 ==> 32 seconds
    
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000]  # periods
    
    # jitters = [0,0.01,0.02,0.05,0.1,0.2,0.5,1]  # maxjitter = percent jitter * period
    jitters = [0,0.02,0.05,0.1,0.2,0.3,0.4,0.5]  # maxjitter = percent jitter * period
    
    # num_chains = [3,5,8,10] 
    num_chains  = [3,5]  # for test
    

    # random_seed = 100  # fixed seed
    # timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime("%Y%m%d_%H%M%S")

    random_seed = int(time.time())
    timestamp = datetime.datetime.fromtimestamp(random_seed).strftime("%Y%m%d_%H%M%S")


    run_results, false_results = run(jitters, num_chains, num_repeats, random_seed, periods)
    
    output_results(num_repeats, random_seed, timestamp, run_results, false_results, num_chains, jitters)
    