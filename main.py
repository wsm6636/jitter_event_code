from evaluation import output_results
from evaluation import generate_periods_and_offsets
from evaluation import filter_and_export_csv
from evaluationC1 import output_results_C1
from evaluationC1 import filter_and_export_csv_C1

from analysis import run_analysis
from analysisC1 import run_analysis_C1

from plot import compare_line_chart_from_csv
from plot import compare_plot_histogram

import random
import datetime
import time
import numpy as np
import os
import sys
import argparse
import pandas as pd

from analysis import Event as ourEvent
from analysis import Task as ourTask
from utilities.task import Task
from mrtanalysis import G2023_analysis
from analysis import run_analysis_for_G2023
from analysisC1 import run_analysis_C1_for_G2023
from evaluation import output_to_csv
from evaluationC1 import output_to_csv_C1
from mrtanalysis import output_to_csv_G2023

from analysis import run_analysis_LET
from analysisC1 import run_analysis_C1_LET
from evaluation import generate_LET
from mrtanalysis import G2023_analysis_LET
from evaluation import output_to_csv_LET
from evaluationC1 import output_to_csv_C1_LET

from plot import compare_plot_histogram_G2023

def convert_to_tasks(num_tasks, selected_periods, schedule_wcet, task_set, schedule_bcet, new_task_set):
    results = []
    selected_read_offsets = []
    selected_write_offsets = []
    read_jitters = []
    write_jitters = []

    for i, (t_bcet, t_wcet) in enumerate(zip(new_task_set, task_set)):
        seq_bcet = schedule_bcet[t_bcet]
        seq_wcet = schedule_wcet[t_wcet]
        Tb = t_bcet.period
        Tw = t_wcet.period
        tr_bcet = [tr - j*Tb for j, (tr, _) in enumerate(seq_bcet)]
        tw_bcet = [tw - j*Tb for j, (_, tw) in enumerate(seq_bcet)]

        tr_wcet = [tr - j*Tw for j, (tr, _) in enumerate(seq_wcet)]
        tw_wcet = [tw - j*Tw for j, (_, tw) in enumerate(seq_wcet)]

        max_tr = max(tr_wcet)
        min_tr = min(tr_bcet)
        max_tw = max(tw_wcet)
        min_tw = min(tw_bcet)

        results=(max_tr, min_tr, max_tw, min_tw)
        read_offset = min_tr
        read_jitter = max_tr - min_tr
        write_offset= min_tw
        write_jitter= max_tw - min_tw

        # print(results)
        selected_read_offsets.append(read_offset)
        read_jitters.append(read_jitter)
        selected_write_offsets.append(write_offset)
        write_jitters.append(write_jitter)

    # print(f"read_jitters{read_jitters}")
    # print(f"write_jitters{write_jitters}")

    return read_jitters, write_jitters, selected_read_offsets, selected_write_offsets



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
                final_e2e_max, max_reaction_time,  final_r, final_w, tasks= run_analysis(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
                
                if final_e2e_max != 0:
                    print(f"final_e2e_max={final_e2e_max}, max_reaction_time={max_reaction_time}")
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
                    print(f"final_e2e_max_C1={final_e2e_max_C1}, max_reaction_time_C1={max_reaction_time_C1}")
                    r_C1 = max_reaction_time_C1 / final_e2e_max_C1
                    if r_C1 > 1 + TOLERANCE:  # if rate is larger than 1, then algorithm failed
                        exceed_C1 = "exceed"
                    else:
                        exceed_C1 = "safe"

                else:
                    r_C1 = None
                    rm_C1 = None
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



def append_to_common_csv(csv_file, common_csv_file):
    try:
        df_current = pd.read_csv(csv_file)
        
        if os.path.exists(common_csv_file):
            df_common = pd.read_csv(common_csv_file)
            df_combined = pd.concat([df_common, df_current], ignore_index=True)
        else:
            df_combined = df_current
        
        df_combined.to_csv(common_csv_file, index=False)
        print(f"Results appended to common CSV: {common_csv_file}")
        
    except Exception as e:
        print(f"Error appending to common CSV: {e}")


def compare_plots(csv_files, num_repeats, random_seed, timestamp):
    folder_name = f"{num_repeats}_{random_seed}_{timestamp}"
    # folder_path = os.path.join("compare/F16", folder_name)
    folder_path = os.path.join("compare/", folder_name)

    os.makedirs(folder_path, exist_ok=True)
    
    compare_percent_plot_name = os.path.join(folder_path,  f"compare_percent_{num_repeats}_{random_seed}_{timestamp}.png")
    compare_histogram_plot_name = os.path.join(folder_path, f"compare_R_{num_repeats}_{random_seed}_{timestamp}.png")

    compare_line_chart_from_csv(csv_files, compare_percent_plot_name)
    compare_plot_histogram(csv_files, compare_histogram_plot_name)
    
    print(f"Compare percent plots generated and saved to {compare_percent_plot_name} and {compare_histogram_plot_name}")



def compare_2023(num_chains, num_repeats, random_seed, periods):

    TOLERANCE = 1e-9
    # preparing list for storing result
    results = {num_tasks: [] for num_tasks in num_chains}
    final = {num_tasks: [] for num_tasks in num_chains}
    false_results = {num_tasks:  0  for num_tasks in num_chains}
    
    results_C1 = {num_tasks:  [] for num_tasks in num_chains}
    final_C1 = {num_tasks:  []  for num_tasks in num_chains}
    false_results_C1 = {num_tasks:  0  for num_tasks in num_chains}

    G2023_results = {num_tasks:  []  for num_tasks in num_chains}

    for i in range(num_repeats):
        random.seed(random_seed)
        for num_tasks in num_chains:
            mrt, let, selected_periods, schedule_wcet, task_set, schedule_bcet, new_task_set,run_time_G = G2023_analysis(num_tasks, periods)
            read_jitters, write_jitters,  selected_read_offsets, selected_write_offsets = convert_to_tasks(num_tasks, selected_periods, schedule_wcet, task_set, schedule_bcet, new_task_set)

            print(f"=========For evaluation========= num_tasks {num_tasks} Repeat {i} random_seed {random_seed} ==================")
            t_our_0 = time.perf_counter()
            final_e2e_max, max_reaction_time,  final_r, final_w, tasks = run_analysis_for_G2023(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, read_jitters, write_jitters )
            t_our_1 = time.perf_counter()
            run_time_our = t_our_1 - t_our_0
            if final_e2e_max != 0:
                # print(f"final_e2e_max={final_e2e_max}, max_reaction_time={max_reaction_time}, mrt={mrt}, let={let}")
                # r = max_reaction_time / final_e2e_max
                r = mrt  / final_e2e_max
                if r > 1 + TOLERANCE:  # if rate is larger than 1, then algorithm failed
                    exceed = "exceed"
                else:
                    exceed = "safe"
            else:
                r = None
                exceed = None
                false_results[num_tasks] += 1  # algorithm failed

            results[num_tasks].append((final_e2e_max, mrt,r,tasks,random_seed,exceed,run_time_our,run_time_G))
            final[num_tasks].append((final_r, final_w))
            G2023_results[num_tasks].append((mrt, let, schedule_bcet, schedule_wcet, random_seed,run_time_G))

            print(f"=========For evaluation C1========= num_tasks {num_tasks} Repeat {i} random_seed {random_seed} ==================")
            t_our_C1_0 = time.perf_counter()
            final_e2e_max_C1, max_reaction_time_C1,  final_r_C1, final_w_C1, tasks_C1, adjust_C1, inserted_C1 = run_analysis_C1_for_G2023(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, read_jitters, write_jitters )
            t_our_C1_1 = time.perf_counter()
            run_time_our_C1 = t_our_C1_1 - t_our_C1_0
            # value of rate "= max_reaction_time / final_e2e_max"
            if final_e2e_max_C1 != 0:
                # print(f"final_e2e_max_C1={final_e2e_max_C1}, max_reaction_time_C1={max_reaction_time_C1}, mrt={mrt}, let={let}")
                # r_C1 = max_reaction_time_C1 / final_e2e_max_C1
                r_C1 = mrt / final_e2e_max_C1
                if r_C1 > 1 + TOLERANCE:  # if rate is larger than 1, then algorithm failed
                    exceed_C1 = "exceed"
                else:
                    exceed_C1 = "safe"
            else:
                r_C1 = None
                exceed_C1 = None
                false_results_C1[num_tasks] += 1  # algorithm failed

            results_C1[num_tasks].append((final_e2e_max_C1, mrt, r_C1, tasks_C1, random_seed, exceed_C1, adjust_C1, inserted_C1,run_time_our_C1,run_time_G))
            final_C1[num_tasks].append((final_r_C1, final_w_C1))


        random_seed += 1

    for num_tasks in num_chains:

        false_percentage = (false_results[num_tasks] / num_repeats)
        false_results[num_tasks]= false_percentage

        false_percentage_C1 = (false_results_C1[num_tasks] / num_repeats)
        false_results_C1[num_tasks] = false_percentage_C1

    return results, false_results, final, results_C1, false_results_C1, final_C1, G2023_results


def compare_2023_LET(jitters, num_chains, num_repeats, random_seed, periods):
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
            selected_periods, selected_read_offsets, selected_write_offsets = generate_LET(num_tasks, periods)
            for per_jitter in jitters:
                print(f"=========For evaluation========= num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                t_our_0 = time.perf_counter()
                final_e2e_max, final_r, final_w, tasks = run_analysis_LET(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
                t_our_1 = time.perf_counter()
                run_time_our = t_our_1 - t_our_0

                t_G_0 = time.perf_counter()
                letG2023 = G2023_analysis_LET(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
                t_G_1 = time.perf_counter()
                run_time_G = t_G_1 - t_G_0

                if final_e2e_max != 0:
                    r = letG2023 / final_e2e_max
                    if r > 1 + TOLERANCE:  # if rate is larger than 1, then algorithm failed
                        exceed = "exceed"
                    else:
                        exceed = "safe"
                else:
                    r = None
                    exceed = None
                    false_results[num_tasks][per_jitter] += 1  # algorithm failed

                results[num_tasks][per_jitter].append((final_e2e_max, letG2023,r,tasks,random_seed,exceed,run_time_our,run_time_G))
                final[num_tasks][per_jitter].append((final_r, final_w))

                print(f"=========For evaluation C1========= num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                t_our_C1_0 = time.perf_counter()
                final_e2e_max_C1, final_r_C1, final_w_C1, tasks_C1, adjust_C1, inserted_C1 = run_analysis_C1_LET(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
                t_our_C1_1 = time.perf_counter()
                run_time_our_C1 = t_our_C1_1 - t_our_C1_0
                # value of rate "= max_reaction_time / final_e2e_max"
                if final_e2e_max_C1 != 0:
                    r_C1 = letG2023 / final_e2e_max_C1
                    if r_C1 > 1 + TOLERANCE:  # if rate is larger than 1, then algorithm failed
                        exceed_C1 = "exceed"
                    else:
                        exceed_C1 = "safe"
                else:
                    r_C1 = None
                    exceed_C1 = None
                    false_results_C1[num_tasks][per_jitter] += 1  # algorithm failed

                results_C1[num_tasks][per_jitter].append((final_e2e_max_C1, letG2023, r_C1, tasks_C1, random_seed, exceed_C1, adjust_C1, inserted_C1, run_time_G,run_time_our_C1))
                final_C1[num_tasks][per_jitter].append((final_r_C1, final_w_C1))


        random_seed += 1

    for num_tasks in num_chains:
        for per_jitter in jitters:
            false_percentage = (false_results[num_tasks][per_jitter] / num_repeats)
            false_results[num_tasks][per_jitter] = false_percentage

            false_percentage_C1 = (false_results_C1[num_tasks][per_jitter] / num_repeats)
            false_results_C1[num_tasks][per_jitter] = false_percentage_C1

    return results, false_results, final, results_C1, false_results_C1, final_C1


def run_MRT_G2023():
    num_repeats = 100000 
    
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000]  # periods
    
    num_chains = [3,5,8,10] 

    # random_seed = 1755016037 # fixed seed
    # timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime("%Y%m%d_%H%M%S")

    random_seed = int(time.time())
    timestamp = datetime.datetime.fromtimestamp(random_seed).strftime("%Y%m%d_%H%M%S")

    results, false_results, final, results_C1, false_results_C1, final_C1, G2023_results = compare_2023(num_chains, num_repeats, random_seed, periods)  

    output_to_csv(num_repeats, random_seed, timestamp, results, false_results, num_chains)
    output_to_csv_C1(num_repeats, random_seed, timestamp, results_C1, false_results_C1, num_chains)
    output_to_csv_G2023(num_repeats, random_seed, timestamp, G2023_results, num_chains)


def run_LET_G2023():
    
    num_repeats = 100000 
    
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000]  # periods
    
    num_chains = [3,5,8,10] 

    # num_repeats = 1 
    
    # periods = [5]  # periods
    
    # num_chains = [2] 

    jitters = [0]
    # random_seed = 1755016037 # fixed seed
    # timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime("%Y%m%d_%H%M%S")

    random_seed = int(time.time())
    timestamp = datetime.datetime.fromtimestamp(random_seed).strftime("%Y%m%d_%H%M%S")

    folder_name = f"{num_repeats}_{random_seed}_{timestamp}"
    # folder_path = os.path.join("compare/F16", folder_name)
    folder_path = os.path.join("compare/", folder_name)

    os.makedirs(folder_path, exist_ok=True)
    
    compare_histogram_plot_name = os.path.join(folder_path, f"compare_G2023_LET_{num_repeats}_{random_seed}_{timestamp}.png")

    results, false_results, final, results_C1, false_results_C1, final_C1 = compare_2023_LET(jitters, num_chains, num_repeats, random_seed, periods)

    csv_file = output_to_csv_LET(num_repeats, random_seed, timestamp, results, false_results, num_chains,jitters)
    csv_fileC1 = output_to_csv_C1_LET(num_repeats, random_seed, timestamp, results_C1, false_results_C1, num_chains,jitters)

    # csv_files = [csv_file, csv_fileC1]
    # compare_plot_histogram_G2023(csv_files,compare_histogram_plot_name)

def main():
    # parser = argparse.ArgumentParser(description='compare C1 and rtsscode')
    # parser.add_argument('random_seed', type=int, help='random seed for the experiment')
    # parser.add_argument('num_repeats', type=int, help='number of repeats for the experiment')
    # parser.add_argument('--common_csv', type=str, default='common_results.csv', 
    #                     help='rtss result csv file (common_results.csv)')
    # parser.add_argument('--common_csv_c1', type=str, default='common_results_c1.csv', 
    #                     help='C1 result csv file (common_results_c1.csv)')
    
    # args = parser.parse_args()
    
    # periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000] 
    # jitters = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5] 
    # num_chains = [3, 5, 8, 10] 
    
    # random_seed = args.random_seed
    # num_repeats = args.num_repeats
    # timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime("%Y%m%d_%H%M%S")
    
    # print(f"random_seed={random_seed}, num_repeats={num_repeats}")
    
    # results, false_results, final, results_C1, false_results_C1, final_C1 = compareC1(
    #     jitters, num_chains, num_repeats, random_seed, periods)

    # csv_file, _, _, _ = output_results(num_repeats, random_seed, timestamp, results, false_results, num_chains, jitters)
    # csv_file_C1, _, _, _ = output_results_C1(num_repeats, random_seed, timestamp, results_C1, false_results_C1, num_chains, jitters)

    
    # append_to_common_csv(csv_file, args.common_csv)
    # append_to_common_csv(csv_file_C1, args.common_csv_c1)
    
    parser = argparse.ArgumentParser(description='compare paperC and rtss')
    parser.add_argument("--LET", action='store_true', help="run LET_G2023 instead of MRT_G2023")

    args = parser.parse_args()

    if args.LET:
        run_LET_G2023()
    else:
        run_MRT_G2023()
    


if __name__ == "__main__":
    main()