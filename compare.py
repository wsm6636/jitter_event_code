from evaluation import output_results
from evaluationF16 import output_results_F16
from evaluationratio import output_results_ratio

from evaluation import generate_periods_and_offsets

from evaluationratio import run_ratio

from analysis import run_analysis
from analysisF16 import run_analysis_F16

import random
import datetime
import time
import numpy as np

def compare(jitters, num_chains, num_repeats, random_seed, periods):
    # preparing list for storing result
    results = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    final = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    false_results = {num_tasks: {per_jitter: 0 for per_jitter in jitters} for num_tasks in num_chains}
    
    # preparing list for storing result F16
    results_F16 = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    final_F16 = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    false_results_F16 = {num_tasks: {per_jitter: 0 for per_jitter in jitters} for num_tasks in num_chains}

    for i in range(num_repeats):
        random.seed(random_seed)
        for num_tasks in num_chains:
            selected_periods, selected_read_offsets, selected_write_offsets = generate_periods_and_offsets(num_tasks, periods)
            for per_jitter in jitters:
                print(f"=========For evaluation========= num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                final_e2e_max, max_reaction_time,  final_r, final_w, tasks = run_analysis(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
                if final_e2e_max != 0:
                    r = max_reaction_time / final_e2e_max
                    if r > 1:
                        exceed = "exceed"
                    else:
                        exceed = "safe"
                else:
                    r = None
                    exceed = None
                    false_results[num_tasks][per_jitter] += 1  # algorithm failed

                results[num_tasks][per_jitter].append((final_e2e_max, max_reaction_time,r,tasks,random_seed,exceed))
                final[num_tasks][per_jitter].append((final_r, final_w))

                print(f"=========For evaluation F16========= num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                final_e2e_max_F16, max_reaction_time_F16,  final_r_F16, final_w_F16, tasks_F16, adjust_F16 = run_analysis_F16(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
                # value of rate "= max_reaction_time / final_e2e_max"
                if final_e2e_max_F16 != 0:
                    r_F16 = max_reaction_time_F16 / final_e2e_max_F16
                    if r_F16 > 1:
                        exceed_F16 = "exceed"
                    else:
                        exceed_F16 = "safe"
                else:
                    r_F16 = None
                    exceed_F16 = None
                    false_results_F16[num_tasks][per_jitter] += 1  # algorithm failed

                results_F16[num_tasks][per_jitter].append((final_e2e_max_F16, max_reaction_time_F16, r_F16, tasks_F16, random_seed, exceed_F16, adjust_F16))
                final_F16[num_tasks][per_jitter].append((final_r_F16, final_w_F16))


        random_seed += 1

    for num_tasks in num_chains:
        for per_jitter in jitters:
            false_percentage = (false_results[num_tasks][per_jitter] / num_repeats)
            false_results[num_tasks][per_jitter] = false_percentage

            false_percentage_F16 = (false_results_F16[num_tasks][per_jitter] / num_repeats)
            false_results_F16[num_tasks][per_jitter] = false_percentage_F16

    return results, false_results, final, results_F16, false_results_F16, final_F16


if __name__ == "__main__":
    # INCREASE here to have more experiments per same settings
    num_repeats = 50
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000] 
    jitters = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5] 
    num_chains = [3,5,8,10] 
    # num_chains = [3, 5]  # for test

    min_period = 1  # minimum period
    max_period = 1000  # maximum period
    ratios = np.arange(1.0, 5.0, 1)
    print(f"Ratios: {ratios}")

    random_seed = 100
    timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime("%Y%m%d_%H%M%S")

    # random_seed = int(time.time())
    # timestamp = datetime.datetime.fromtimestamp(random_seed).strftime("%Y%m%d_%H%M%S")

    results, false_results, final, results_F16, false_results_F16, final_F16 = compare(jitters, num_chains, num_repeats, random_seed, periods)
    

    output_results(num_repeats, random_seed, timestamp, results, false_results, num_chains, jitters)
    output_results_F16(num_repeats, random_seed, timestamp, results_F16, false_results_F16, num_chains, jitters)

    print("===================RATIO========================")
    results_ratio, false_results_ratio, final_task_ratio = run_ratio(jitters, num_chains, num_repeats, random_seed, ratios, min_period, max_period)
    output_results_ratio(num_repeats, random_seed, timestamp, results_ratio, false_results_ratio, num_chains, jitters, ratios)


