from evaluation import output_results
from evaluationF16 import output_results_F16
from evaluationRW import output_results_RW
from evaluation import generate_periods_and_offsets
from evaluationLET import output_results_LET
from evaluationC1 import output_results_C1
from evaluationratio import output_results_ratio
from evaluationratio import run_ratio

from analysis import run_analysis
from analysisF16 import run_analysis_F16
from analysisRW import run_analysis_RW
from analysisLET import run_analysis_LET
from analysisC1 import run_analysis_C1

from plot import compare_line_chart_from_csv
from plot import compare_plot_histogram
from plot import comparetre_line_chart_from_csv
from plot import comparetre_plot_histogram

import random
import datetime
import time
import numpy as np
import os



def compareF16(jitters, num_chains, num_repeats, random_seed, periods):
    TOLERANCE = 1e-9
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

                print(f"=========For evaluation F16========= num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                final_e2e_max_F16, max_reaction_time_F16,  final_r_F16, final_w_F16, tasks_F16, adjust_F16 = run_analysis_F16(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
                # value of rate "= max_reaction_time / final_e2e_max"
                if final_e2e_max_F16 != 0:
                    r_F16 = max_reaction_time_F16 / final_e2e_max_F16
                    if r_F16 > 1 + TOLERANCE:  # if rate is larger than 1, then algorithm failed
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



def compareRW(jitters, num_chains, num_repeats, random_seed, periods):
    TOLERANCE = 1e-9
    # preparing list for storing result
    results = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    false_results = {num_tasks: {per_jitter: 0 for per_jitter in jitters} for num_tasks in num_chains}
    
    # preparing list for storing result F16
    results_RW = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    false_results_RW = {num_tasks: {per_jitter: 0 for per_jitter in jitters} for num_tasks in num_chains}

    for i in range(num_repeats):
        random.seed(random_seed)
        for num_tasks in num_chains:
            selected_periods, selected_read_offsets, selected_write_offsets = generate_periods_and_offsets(num_tasks, periods)
            for per_jitter in jitters:
                print(f"=========For evaluation========= num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                final_e2e_max, max_reaction_time,  _,_, tasks = run_analysis(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
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

                print(f"=========For evaluation RW========= num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                final_e2e_max_RW, max_reaction_time_RW,  tasks_RW = run_analysis_RW(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
                # value of rate "= max_reaction_time / final_e2e_max"
                if final_e2e_max_RW != 0:
                    r_RW = max_reaction_time_RW / final_e2e_max_RW
                    if r_RW > 1 + TOLERANCE:  # if rate is larger than 1, then algorithm failed
                        exceed_RW = "exceed"
                    else:
                        exceed_RW = "safe"
                else:
                    r_RW = None
                    exceed_RW = None
                    false_results_RW[num_tasks][per_jitter] += 1  # algorithm failed

                results_RW[num_tasks][per_jitter].append((final_e2e_max_RW, max_reaction_time_RW, r_RW, tasks_RW, random_seed, exceed_RW))

        random_seed += 1

    for num_tasks in num_chains:
        for per_jitter in jitters:
            false_percentage = (false_results[num_tasks][per_jitter] / num_repeats)
            false_results[num_tasks][per_jitter] = false_percentage

            false_percentage_RW = (false_results_RW[num_tasks][per_jitter] / num_repeats)
            false_results_RW[num_tasks][per_jitter] = false_percentage_RW

    return results, false_results, results_RW, false_results_RW


def compareLET(jitters, num_chains, num_repeats, random_seed, periods):
    TOLERANCE = 1e-9
    # preparing list for storing result
    results = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    final = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    false_results = {num_tasks: {per_jitter: 0 for per_jitter in jitters} for num_tasks in num_chains}
    
    # preparing list for storing result LET
    results_LET = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    final_LET = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    false_results_LET = {num_tasks: {per_jitter: 0 for per_jitter in jitters} for num_tasks in num_chains}

    for i in range(num_repeats):
        random.seed(random_seed)
        for num_tasks in num_chains:
            selected_periods, selected_read_offsets, selected_write_offsets = generate_periods_and_offsets(num_tasks, periods)
            for per_jitter in jitters:
                print(f"=========For evaluation========= num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                final_e2e_max, max_reaction_time,  final_r, final_w, tasks = run_analysis(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
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

                print(f"=========For evaluation LET========= num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                final_e2e_max_LET, max_reaction_time_LET,  final_r_LET, final_w_LET, tasks_LET, inserted_LET = run_analysis_LET(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
                # value of rate "= max_reaction_time / final_e2e_max"
                if final_e2e_max_LET != 0:
                    r_LET = max_reaction_time_LET / final_e2e_max_LET
                    if r_LET > 1 + TOLERANCE:  # if rate is larger than 1, then algorithm failed
                        exceed_LET = "exceed"
                    else:
                        exceed_LET = "safe"
                else:
                    r_LET = None
                    exceed_LET = None
                    false_results_LET[num_tasks][per_jitter] += 1  # algorithm failed

                results_LET[num_tasks][per_jitter].append((final_e2e_max_LET, max_reaction_time_LET, r_LET, tasks_LET, random_seed, exceed_LET, inserted_LET))
                final_LET[num_tasks][per_jitter].append((final_r_LET, final_w_LET))


        random_seed += 1

    for num_tasks in num_chains:
        for per_jitter in jitters:
            false_percentage = (false_results[num_tasks][per_jitter] / num_repeats)
            false_results[num_tasks][per_jitter] = false_percentage

            false_percentage_LET = (false_results_LET[num_tasks][per_jitter] / num_repeats)
            false_results_LET[num_tasks][per_jitter] = false_percentage_LET

    return results, false_results, final, results_LET, false_results_LET, final_LET


def compareC1(jitters, num_chains, num_repeats, random_seed, periods):
    TOLERANCE = 1e-9
    # preparing list for storing result
    results = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    final = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    false_results = {num_tasks: {per_jitter: 0 for per_jitter in jitters} for num_tasks in num_chains}
    
    results_C1 = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    final_C1 = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    false_results_C1 = {num_tasks: {per_jitter: 0 for per_jitter in jitters} for num_tasks in num_chains}

    for i in range(num_repeats):
        random.seed(random_seed)
        for num_tasks in num_chains:
            selected_periods, selected_read_offsets, selected_write_offsets = generate_periods_and_offsets(num_tasks, periods)
            for per_jitter in jitters:
                print(f"=========For evaluation========= num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                final_e2e_max, max_reaction_time,  final_r, final_w, tasks = run_analysis(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
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

                print(f"=========For evaluation C1========= num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                final_e2e_max_C1, max_reaction_time_C1,  final_r_C1, final_w_C1, tasks_C1, adjust_C1, inserted_C1 = run_analysis_C1(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
                # value of rate "= max_reaction_time / final_e2e_max"
                if final_e2e_max_C1 != 0:
                    r_C1 = max_reaction_time_C1 / final_e2e_max_C1
                    if r_C1 > 1 + TOLERANCE:  # if rate is larger than 1, then algorithm failed
                        exceed_C1 = "exceed"
                    else:
                        exceed_C1 = "safe"
                else:
                    r_C1 = None
                    exceed_C1 = None
                    false_results_C1[num_tasks][per_jitter] += 1  # algorithm failed

                results_C1[num_tasks][per_jitter].append((final_e2e_max_C1, max_reaction_time_C1, r_C1, tasks_C1, random_seed, exceed_C1, adjust_C1, inserted_C1))
                final_C1[num_tasks][per_jitter].append((final_r_C1, final_w_C1))


        random_seed += 1

    for num_tasks in num_chains:
        for per_jitter in jitters:
            false_percentage = (false_results[num_tasks][per_jitter] / num_repeats)
            false_results[num_tasks][per_jitter] = false_percentage

            false_percentage_C1 = (false_results_C1[num_tasks][per_jitter] / num_repeats)
            false_results_C1[num_tasks][per_jitter] = false_percentage_C1

    return results, false_results, final, results_C1, false_results_C1, final_C1



def compareTRE(jitters, num_chains, num_repeats, random_seed, periods):
    TOLERANCE = 1e-9
    # preparing list for storing result
    results = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    final = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    false_results = {num_tasks: {per_jitter: 0 for per_jitter in jitters} for num_tasks in num_chains}
    
    results_C1 = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    final_C1 = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    false_results_C1 = {num_tasks: {per_jitter: 0 for per_jitter in jitters} for num_tasks in num_chains}

    results_LET = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    final_LET = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    false_results_LET = {num_tasks: {per_jitter: 0 for per_jitter in jitters} for num_tasks in num_chains}

    for i in range(num_repeats):
        random.seed(random_seed)
        for num_tasks in num_chains:
            selected_periods, selected_read_offsets, selected_write_offsets = generate_periods_and_offsets(num_tasks, periods)
            for per_jitter in jitters:
                print(f"=========For evaluation========= num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                final_e2e_max, max_reaction_time,  final_r, final_w, tasks = run_analysis(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
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

                print(f"=========For evaluation C1========= num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                final_e2e_max_C1, max_reaction_time_C1,  final_r_C1, final_w_C1, tasks_C1, adjust_C1, inserted_C1 = run_analysis_C1(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
                # value of rate "= max_reaction_time / final_e2e_max"
                if final_e2e_max_C1 != 0:
                    r_C1 = max_reaction_time_C1 / final_e2e_max_C1
                    if r_C1 > 1 + TOLERANCE:  # if rate is larger than 1, then algorithm failed
                        exceed_C1 = "exceed"
                    else:
                        exceed_C1 = "safe"
                else:
                    r_C1 = None
                    exceed_C1 = None
                    false_results_C1[num_tasks][per_jitter] += 1  # algorithm failed

                results_C1[num_tasks][per_jitter].append((final_e2e_max_C1, max_reaction_time_C1, r_C1, tasks_C1, random_seed, exceed_C1, adjust_C1, inserted_C1))
                final_C1[num_tasks][per_jitter].append((final_r_C1, final_w_C1))

                print(f"=========For evaluation LET========= num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                final_e2e_max_LET, max_reaction_time_LET,  final_r_LET, final_w_LET, tasks_LET, inserted_LET = run_analysis_LET(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
                # value of rate "= max_reaction_time / final_e2e_max"
                if final_e2e_max_LET != 0:
                    r_LET = max_reaction_time_LET / final_e2e_max_LET
                    if r_LET > 1 + TOLERANCE:  # if rate is larger than 1, then algorithm failed
                        exceed_LET = "exceed"
                    else:
                        exceed_LET = "safe"
                else:
                    r_LET = None
                    exceed_LET = None
                    false_results_LET[num_tasks][per_jitter] += 1  # algorithm failed

                results_LET[num_tasks][per_jitter].append((final_e2e_max_LET, max_reaction_time_LET, r_LET, tasks_LET, random_seed, exceed_LET, inserted_LET))
                final_LET[num_tasks][per_jitter].append((final_r_LET, final_w_LET))

        random_seed += 1

    for num_tasks in num_chains:
        for per_jitter in jitters:
            false_percentage = (false_results[num_tasks][per_jitter] / num_repeats)
            false_results[num_tasks][per_jitter] = false_percentage

            false_percentage_C1 = (false_results_C1[num_tasks][per_jitter] / num_repeats)
            false_results_C1[num_tasks][per_jitter] = false_percentage_C1

            false_percentage_LET = (false_results_LET[num_tasks][per_jitter] / num_repeats)
            false_results_LET[num_tasks][per_jitter] = false_percentage_LET

    return results, false_results, final, results_C1, false_results_C1, final_C1, results_LET, false_results_LET, final_LET


def compare_plots(csv_files, num_repeats, random_seed, timestamp):
    folder_name = f"{num_repeats}_{random_seed}_{timestamp}"
    # folder_path = os.path.join("compare/F16", folder_name)
    folder_path = os.path.join("compare/TRE", folder_name)

    os.makedirs(folder_path, exist_ok=True)
    
    compare_percent_plot_name = os.path.join(folder_path,  f"compare_percent_{num_repeats}_{random_seed}_{timestamp}.png")
    compare_histogram_plot_name = os.path.join(folder_path, f"compare_R_{num_repeats}_{random_seed}_{timestamp}.png")

    # compare_line_chart_from_csv(csv_files, compare_percent_plot_name)
    # compare_plot_histogram(csv_files, compare_histogram_plot_name)
    comparetre_line_chart_from_csv(csv_files, compare_percent_plot_name)
    comparetre_plot_histogram(csv_files, compare_histogram_plot_name)
    
    print(f"Compare percent plots generated and saved to {compare_percent_plot_name} and {compare_histogram_plot_name}")


if __name__ == "__main__":
    # INCREASE here to have more experiments per same settings
    num_repeats =  10
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000] 
    jitters = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5] 
    num_chains = [3,5,8,10] 
    # num_chains = [3, 5]  # for test

    min_period = 1  # minimum period
    ratios = np.arange(1.0, 6.0, 0.5)
    print(f"Ratios: {ratios}")

    # random_seed = 1754657734
    # timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime("%Y%m%d_%H%M%S")

    random_seed = int(time.time())
    timestamp = datetime.datetime.fromtimestamp(random_seed).strftime("%Y%m%d_%H%M%S")

    # results, false_results, final, results_F16, false_results_F16, final_F16 = compareF16(jitters, num_chains, num_repeats, random_seed, periods)
    # results, false_results, results_RW, false_results_RW = compareRW(jitters, num_chains, num_repeats, random_seed, periods)
    # results, false_results, final, results_LET, false_results_LET, final_LET = compareLET(jitters, num_chains, num_repeats, random_seed, periods)
    # results, false_results, final, results_C1, false_results_C1, final_C1 = compareC1(jitters, num_chains, num_repeats, random_seed, periods)
    results, false_results, final, results_C1, false_results_C1, final_C1, results_LET, false_results_LET, final_LET = compareTRE(jitters, num_chains, num_repeats, random_seed, periods)

    csv_file, _, _, _ = output_results(num_repeats, random_seed, timestamp, results, false_results, num_chains, jitters)
    # csv_file_F16, _, _, _ = output_results_F16(num_repeats, random_seed, timestamp, results_F16, false_results_F16, num_chains, jitters)
    # csv_file_RW, _, _, _ = output_results_RW(num_repeats, random_seed, timestamp, results_RW, false_results_RW, num_chains, jitters)
    csv_file_LET, _, _, _ = output_results_LET(num_repeats, random_seed, timestamp, results_LET, false_results_LET, num_chains, jitters)
    csv_file_C1, _, _, _ = output_results_C1(num_repeats, random_seed, timestamp, results_C1, false_results_C1, num_chains, jitters)

    # csv_files = [csv_file, csv_file_F16]
    # csv_files = [csv_file, csv_file_RW]
    # csv_files = [csv_file, csv_file_LET]
    # csv_files = [csv_file, csv_file_C1]
    csv_files = [csv_file, csv_file_LET, csv_file_C1]

    compare_plots(csv_files, num_repeats, random_seed, timestamp)

    # print("===================RATIO========================")

    # ratio_num_repeats = 10000
    # results_ratio, false_results_ratio, final_task_ratio = run_ratio(jitters, num_chains, ratio_num_repeats, random_seed, ratios, min_period)
    # output_results_ratio(ratio_num_repeats, random_seed, timestamp, results_ratio, false_results_ratio, num_chains, jitters, ratios)


    