from evaluation import output_our
from evaluation import generate_periods_and_offsets
from evaluation import output_G2023_MRT
from evaluation import generate_LET
from evaluation import output_G2023_LET
from evaluation_adjust import output_adjust
from evaluation_adjust import output_adjust_G2023_MRT

from analysis import run_analysis_our
from analysis import run_analysis_our_for_G2023_LET
from analysis import run_analysis_our_for_G2023_MRT
from analysis_adjust import run_analysis_adjust
from analysis_adjust import run_analysis_adjust_for_G2023_LET
from analysis_adjust import run_analysis_adjust_for_G2023_MRT
from evaluation_adjust import output_adjust_G2023_LET

from analysis_G2023 import G2023_analysis_MRT
from analysis_G2023 import G2023_analysis_LET

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

"""Extract the offset and jitter that match our paper from the G2023 schedule"""
def convert_to_our_offsets(schedule_wcet, task_set, schedule_bcet, new_task_set):
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

        read_offset = min_tr
        read_jitter = max_tr - min_tr
        write_offset= min_tw
        write_jitter= max_tw - min_tw

        selected_read_offsets.append(read_offset)
        read_jitters.append(read_jitter)
        selected_write_offsets.append(write_offset)
        write_jitters.append(write_jitter)
        
    return read_jitters, write_jitters, selected_read_offsets, selected_write_offsets


"""Experiments 1 and 2(adjusted task chain) of RTSS Paper 2025"""
def compare_our_and_adjust(jitters, num_chains, num_repeats, random_seed, periods):
    TOLERANCE = 1e-9
    # preparing list for storing result
    results = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    final = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    false_results = {num_tasks: {per_jitter: 0 for per_jitter in jitters} for num_tasks in num_chains}
    
    results_adjust = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    final_adjust = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    false_results_adjust = {num_tasks: {per_jitter: 0 for per_jitter in jitters} for num_tasks in num_chains}
    print(f"randomseed{random_seed}")
    for i in range(num_repeats):
        random.seed(random_seed)
        print(f"randomseed{random_seed}")
        for num_tasks in num_chains:
            selected_periods, selected_read_offsets, selected_write_offsets = generate_periods_and_offsets(num_tasks, periods)
            for per_jitter in jitters:
                print(f"=========For evaluation========= num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                final_e2e_max, max_reaction_time,  final_r, final_w, tasks= run_analysis_our(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
                
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

                print(f"=========For evaluation adjust========= num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                final_e2e_max_adjust, max_reaction_time_adjust,  final_r_adjust, final_w_adjust, tasks_adjust, adjust_adjust, inserted_adjust = run_analysis_adjust(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
                # value of rate "= max_reaction_time / final_e2e_max"
                if final_e2e_max_adjust != 0:
                    print(f"final_e2e_max_adjust={final_e2e_max_adjust}, max_reaction_time_adjust={max_reaction_time_adjust}")
                    r_adjust = max_reaction_time_adjust / final_e2e_max_adjust
                    if r_adjust > 1 + TOLERANCE:  # if rate is larger than 1, then algorithm failed
                        exceed_adjust = "exceed"
                    else:
                        exceed_adjust = "safe"

                else:
                    r_adjust = None
                    rm_adjust = None
                    exceed_adjust = None
                    false_results_adjust[num_tasks][per_jitter] += 1  # algorithm failed

                results_adjust[num_tasks][per_jitter].append((final_e2e_max_adjust, max_reaction_time_adjust, r_adjust, tasks_adjust, random_seed, exceed_adjust, adjust_adjust, inserted_adjust))
                final_adjust[num_tasks][per_jitter].append((final_r_adjust, final_w_adjust))


        random_seed += 1

    for num_tasks in num_chains:
        for per_jitter in jitters:
            false_percentage = (false_results[num_tasks][per_jitter] / num_repeats)
            false_results[num_tasks][per_jitter] = false_percentage

            false_percentage_adjust = (false_results_adjust[num_tasks][per_jitter] / num_repeats)
            false_results_adjust[num_tasks][per_jitter] = false_percentage_adjust

    return results, false_results, final, results_adjust, false_results_adjust, final_adjust


"""Comparison of RTSS2025(our, adjust) and G2023 MRT
    G2023: [C] Günzel, Mario, et al. "Compositional timing analysis of asynchronized distributed cause-effect chains." ACM Transactions on Embedded Computing Systems 22.4 (2023): 1-34.
"""
def compare_G2023_MRT(num_chains, num_repeats, random_seed, periods):

    TOLERANCE = 1e-9
    # preparing list for storing result
    results = {num_tasks: [] for num_tasks in num_chains}
    final = {num_tasks: [] for num_tasks in num_chains}
    false_results = {num_tasks:  0  for num_tasks in num_chains}
    
    results_adjust = {num_tasks:  [] for num_tasks in num_chains}
    final_adjust = {num_tasks:  []  for num_tasks in num_chains}
    false_results_adjust = {num_tasks:  0  for num_tasks in num_chains}

    for i in range(num_repeats):
        random.seed(random_seed)
        for num_tasks in num_chains:
            mrt, selected_periods, schedule_wcet, task_set, schedule_bcet, new_task_set,run_time_G = G2023_analysis_MRT(num_tasks, periods)
            print(selected_periods)
            read_jitters, write_jitters,  selected_read_offsets, selected_write_offsets = convert_to_our_offsets(schedule_wcet, task_set, schedule_bcet, new_task_set)
            print(f"=========For evaluation adjust & G2023MRT========= num_tasks {num_tasks} Repeat {i} random_seed {random_seed} ==================")
            t_our_0 = time.perf_counter()
            final_e2e_max, final_r, final_w, tasks = run_analysis_our_for_G2023_MRT(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, read_jitters, write_jitters )
            t_our_1 = time.perf_counter()
            run_time_our = t_our_1 - t_our_0
            if final_e2e_max != 0:
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

            print(f"=========For evaluation adjust & G2023MRT========= num_tasks {num_tasks} Repeat {i} random_seed {random_seed} ==================")
            t_our_adjust_0 = time.perf_counter()
            final_e2e_max_adjust, final_r_adjust, final_w_adjust, tasks_adjust, adjust_adjust, inserted_adjust = run_analysis_adjust_for_G2023_MRT(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, read_jitters, write_jitters )
            t_our_adjust_1 = time.perf_counter()
            run_time_our_adjust = t_our_adjust_1 - t_our_adjust_0
            if final_e2e_max_adjust != 0:
                # r_adjust = max_reaction_time_adjust / final_e2e_max_adjust
                r_adjust = mrt / final_e2e_max_adjust
                if r_adjust > 1 + TOLERANCE:  # if rate is larger than 1, then algorithm failed
                    exceed_adjust = "exceed"
                else:
                    exceed_adjust = "safe"
            else:
                r_adjust = None
                exceed_adjust = None
                false_results_adjust[num_tasks] += 1  # algorithm failed

            results_adjust[num_tasks].append((final_e2e_max_adjust, mrt, r_adjust, tasks_adjust, random_seed, exceed_adjust, adjust_adjust, inserted_adjust,run_time_our_adjust,run_time_G))
            final_adjust[num_tasks].append((final_r_adjust, final_w_adjust))


        random_seed += 1

    for num_tasks in num_chains:

        false_percentage = (false_results[num_tasks] / num_repeats)
        false_results[num_tasks]= false_percentage

        false_percentage_adjust = (false_results_adjust[num_tasks] / num_repeats)
        false_results_adjust[num_tasks] = false_percentage_adjust

    return results, false_results, final, results_adjust, false_results_adjust, final_adjust


"""Comparison of RTSS2025(our, adjust) and G2023 LET"""
def compare_G2023_LET(jitters, num_chains, num_repeats, random_seed, periods):
    TOLERANCE = 1e-9
    # preparing list for storing result
    results = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    final = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    false_results = {num_tasks: {per_jitter: 0 for per_jitter in jitters} for num_tasks in num_chains}
    
    results_adjust = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    final_adjust = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    false_results_adjust = {num_tasks: {per_jitter: 0 for per_jitter in jitters} for num_tasks in num_chains}
    
    for i in range(num_repeats):
        random.seed(random_seed)
        print(f"randomseed{random_seed}")
        for num_tasks in num_chains:
            selected_periods, selected_read_offsets, selected_write_offsets = generate_LET(num_tasks, periods)
            for per_jitter in jitters:
                print(f"=========For evaluation adjust & G2023LET========= num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                t_our_0 = time.perf_counter()
                final_e2e_max, final_r, final_w, tasks = run_analysis_our_for_G2023_LET(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
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

                print(f"=========For evaluation adjust & G2023LET========= num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                t_our_adjust_0 = time.perf_counter()
                final_e2e_max_adjust, final_r_adjust, final_w_adjust, tasks_adjust, adjust_adjust, inserted_adjust = run_analysis_adjust_for_G2023_LET(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
                t_our_adjust_1 = time.perf_counter()
                run_time_our_adjust = t_our_adjust_1 - t_our_adjust_0
                # value of rate "= max_reaction_time / final_e2e_max"
                if final_e2e_max_adjust != 0:
                    r_adjust = letG2023 / final_e2e_max_adjust
                    if r_adjust > 1 + TOLERANCE:  # if rate is larger than 1, then algorithm failed
                        exceed_adjust = "exceed"
                    else:
                        exceed_adjust = "safe"
                else:
                    r_adjust = None
                    exceed_adjust = None
                    false_results_adjust[num_tasks][per_jitter] += 1  # algorithm failed

                results_adjust[num_tasks][per_jitter].append((final_e2e_max_adjust, letG2023, r_adjust, tasks_adjust, random_seed, exceed_adjust, adjust_adjust, inserted_adjust, run_time_G,run_time_our_adjust))
                final_adjust[num_tasks][per_jitter].append((final_r_adjust, final_w_adjust))

        random_seed += 1

    for num_tasks in num_chains:
        for per_jitter in jitters:
            false_percentage = (false_results[num_tasks][per_jitter] / num_repeats)
            false_results[num_tasks][per_jitter] = false_percentage

            false_percentage_adjust = (false_results_adjust[num_tasks][per_jitter] / num_repeats)
            false_results_adjust[num_tasks][per_jitter] = false_percentage_adjust

    return results, false_results, final, results_adjust, false_results_adjust, final_adjust


"""append tmp.csv to common_csv"""
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


"""Comparison plots(line,hist) of our and adjust"""
def compare_plots(csv_files, num_repeats, random_seed, timestamp):
    folder_name = f"{num_repeats}_{random_seed}_{timestamp}"
    folder_path = os.path.join("compare/", folder_name)

    os.makedirs(folder_path, exist_ok=True)
    
    compare_percent_plot_name = os.path.join(folder_path,  f"compare_percent_{num_repeats}_{random_seed}_{timestamp}.png")
    compare_histogram_plot_name = os.path.join(folder_path, f"compare_R_{num_repeats}_{random_seed}_{timestamp}.png")

    compare_line_chart_from_csv(csv_files, compare_percent_plot_name)
    compare_plot_histogram(csv_files, compare_histogram_plot_name)
    
    print(f"Compare percent plots generated and saved to {compare_percent_plot_name} and {compare_histogram_plot_name}")


"""Implicit communication comparison experiment (our/adjust/MRT)"""
def run_G2023_MRT(random_seed, num_repeats, common_csv, common_csv_adjust):
# def run_G2023_MRT():
#     num_repeats = 1
    
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000]  # periods
    
    num_chains = [3,5,8,10] 
    # random_seed = 1755024332
    timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime("%Y%m%d_%H%M%S")

    results, false_results, final, results_adjust, false_results_adjust, final_adjust = compare_G2023_MRT(num_chains, num_repeats, random_seed, periods)  

    csv_file = output_G2023_MRT(num_repeats, random_seed, timestamp, results, false_results, num_chains)
    csv_file_adjust = output_adjust_G2023_MRT(num_repeats, random_seed, timestamp, results_adjust, false_results_adjust, num_chains)
    append_to_common_csv(csv_file, common_csv)
    append_to_common_csv(csv_file_adjust, common_csv_adjust)


"""LET communication comparison experiment (our/adjust/LET)"""
def run_G2023_LET(random_seed, num_repeats, common_csv, common_csv_adjust):    
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000]  # periods
    
    num_chains = [3,5,8,10] 

    jitters = [0]
    timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime("%Y%m%d_%H%M%S")

    results, false_results, final, results_adjust, false_results_adjust, final_adjust = compare_G2023_LET(jitters, num_chains, num_repeats, random_seed, periods)

    csv_file = output_G2023_LET(num_repeats, random_seed, timestamp, results, false_results, num_chains,jitters)
    csv_file_adjust = output_adjust_G2023_LET(num_repeats, random_seed, timestamp, results_adjust, false_results_adjust, num_chains,jitters)

    append_to_common_csv(csv_file, common_csv)
    append_to_common_csv(csv_file_adjust, common_csv_adjust)


"""LET(jitter=0)/Implicit communication comparison experiment (our/adjust)"""
def run_our_and_adjusted(random_seed, num_repeats, common_csv, common_csv_adjust):
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000] 
    jitters = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5] 
    num_chains = [3, 5, 8, 10] 
    
    timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime("%Y%m%d_%H%M%S")
        
    results, false_results, final, results_adjust, false_results_adjust, final_adjust = compare_our_and_adjust(
        jitters, num_chains, num_repeats, random_seed, periods)

    csv_file, _, _, _ = output_our(num_repeats, random_seed, timestamp, results, false_results, num_chains, jitters)
    csv_file_adjust, _, _, _ = output_adjust(num_repeats, random_seed, timestamp, results_adjust, false_results_adjust, num_chains, jitters)
    append_to_common_csv(csv_file, common_csv)
    append_to_common_csv(csv_file_adjust, common_csv_adjust)


def main():
    parser = argparse.ArgumentParser(description='compare our, our_adjust, G2023MRT, G2023LET')
    parser.add_argument('random_seed', type=int, help='random seed for the experiment')
    parser.add_argument('num_repeats', type=int, help='number of repeats for the experiment')
    parser.add_argument('--common_csv', type=str, default='common_results.csv', 
                        help='rtss result csv file (common_results.csv)')
    parser.add_argument('--common_csv_adjust', type=str, default='common_results_adjust.csv', 
                        help='adjust result csv file (common_results_adjust.csv)')
    parser.add_argument('--suffix', default='', help='different name like _MRT/_LET (RTSS)')
    parser.add_argument('--alg', choices=['MRT', 'LET', 'RTSS'], default='RTSS',
                    help='which algorithm to run')
    args = parser.parse_args()
    
    

    alg_map = {
        'MRT':  run_G2023_MRT,
        'LET':  run_G2023_LET,
        'RTSS': run_our_and_adjusted,
    }


    random_seed = args.random_seed
    num_repeats = args.num_repeats
    common_csv   = args.common_csv
    common_csv_adjust = args.common_csv_adjust

    alg = args.alg
    alg_map[alg](random_seed, num_repeats, common_csv, common_csv_adjust)

    # run_G2023_MRT()    

if __name__ == "__main__":
    main()