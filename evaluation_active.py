# 文件名：main.py
import csv
import datetime
from analysis_active import run_analysis_active_our
import random
import time
import os
from evaluation_passive import generate_periods_and_offsets


def output_active_Gunzel_IC(num_repeats, random_seed, timestamp, results, false_results, num_chains):
    folder_path = "active"
    os.makedirs(folder_path, exist_ok=True)

    results_csv = os.path.join(folder_path, f"data_active_IC_{num_repeats}_{random_seed}_{timestamp}.csv" )

    # save results to csv
    with open(results_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["seeds","num_tasks", "final_e2e_max", "ic", "R", "exceed", "false_percentage", "adjusted", "inserted","run_time_our","run_time_G"])
        for num_tasks in num_chains:
            false_percentage = false_results[num_tasks]
            for (final_e2e_max, ic, r, tasks, seed, exceed,adjusted,inserted,run_time_our,run_time_G) in results[num_tasks]:
                writer.writerow([seed,num_tasks, final_e2e_max, ic, r, exceed, false_percentage,adjusted,inserted,run_time_our,run_time_G])

    print(f"All results saved to {results_csv}")

    return results_csv


def output_active_Gunzel_LET(num_repeats, random_seed, timestamp, results, false_results, num_chains, jitters):
    folder_path = "active"
    os.makedirs(folder_path, exist_ok=True)

    results_csv = os.path.join(folder_path, f"data_active_LET_{num_repeats}_{random_seed}_{timestamp}.csv" )
    # save results to csv
    with open(results_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["seeds","num_tasks", "final_e2e_max", "let", "R", "exceed", "false_percentage", "adjusted", "inserted","run_time_G","run_time_our"])
        for num_tasks in num_chains:
            for per_jitter in jitters:
                false_percentage = false_results[num_tasks][per_jitter]
                for (final_e2e_max, let, r, tasks, seed, exceed,adjusted,inserted,run_time_G,run_time_our) in results[num_tasks][per_jitter]:
                    writer.writerow([seed,num_tasks, final_e2e_max, let, r, exceed, false_percentage,adjusted,inserted,run_time_G,run_time_our])

    print(f"All results saved to {results_csv}")

    return results_csv


def output_active_our(num_repeats, random_seed, timestamp, results, false_results, num_chains, jitters):

    folder_path = "active"
    os.makedirs(folder_path, exist_ok=True)

    percent_plot_name = os.path.join(folder_path,  f"percent_active_{num_repeats}_{random_seed}_{timestamp}.png")
    R_plot_name = os.path.join(folder_path, f"R_active_{num_repeats}_{random_seed}_{timestamp}.png")
    results_csv = os.path.join(folder_path, f"data_active_{num_repeats}_{random_seed}_{timestamp}.csv" )

    # save results to csv
    with open(results_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["seeds","num_tasks", "per_jitter", "final_e2e_max", "max_reaction_time", "R", "exceed", "false_percentage", "adjusted", "inserted"])
        for num_tasks in num_chains:
            for per_jitter in jitters:
                false_percentage = false_results[num_tasks][per_jitter]
                for (final_e2e_max, max_reaction_time, r, tasks, seed, exceed,adjusted,inserted) in results[num_tasks][per_jitter]:
                    writer.writerow([seed,num_tasks, per_jitter, final_e2e_max, max_reaction_time, r, exceed, false_percentage,adjusted,inserted])

    print(f"All results saved to {results_csv}")

    return results_csv, percent_plot_name, R_plot_name



def run_evaluation_active_our(jitters, num_chains, num_repeats, random_seed, periods):
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
                final_e2e_max, max_reaction_time,  final_r, final_w, tasks, adjusted, inserted = run_analysis_active_our(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
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

                results[num_tasks][per_jitter].append((final_e2e_max, max_reaction_time,r,tasks,random_seed,exceed,adjusted,inserted))
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
    num_repeats = 1  
    
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000]  # periods
    
    jitters = [0,0.02,0.05,0.1,0.2,0.3,0.4,0.5]  # maxjitter = percent jitter * period

    # num_chains = [3,5,8,10] 
    num_chains  = [3,5]  # for test

    random_seed = 1754657734
    timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime("%Y%m%d_%H%M%S")

    # random_seed = int(time.time())
    # timestamp = datetime.datetime.fromtimestamp(random_seed).strftime("%Y%m%d_%H%M%S")

    run_evaluation_active_our(jitters, num_chains, num_repeats, random_seed, periods)


