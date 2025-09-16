#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 05 10:25:52 2025

It implements the methods described in the paper
    Shumo Wang, Enrico Bini, Martina Maggio, Qingxu Deng
    "Jitter Propagation in Task Chains"

@author: Shumo Wang
"""

from evaluation_passive import generate_periods_and_offsets
from evaluation_passive import generate_LET

from evaluation_passive import output_passive_our
from evaluation_passive import output_passive_Gunzel_IC
from evaluation_passive import output_passive_Gunzel_LET
from evaluation_active import output_active_our
from evaluation_active import output_active_Gunzel_IC
from evaluation_active import output_active_Gunzel_LET

from analysis_passive import run_analysis_passive_our
from analysis_passive import run_analysis_passive_Gunzel_LET
from analysis_passive import run_analysis_passive_Gunzel_IC
from analysis_active import run_analysis_active_our
from analysis_active import run_analysis_active_Gunzel_LET
from analysis_active import run_analysis_active_Gunzel_IC

from analysis_Gunzel import run_analysis_Gunzel_IC
from analysis_Gunzel import run_analysis_Gunzel_LET

from plot import compare_false_percent_our
from plot import compare_plot_histogram_our

import random
import datetime
import time
import os
import argparse
import pandas as pd


def convert_to_our_offsets(schedule_wcet, task_set, schedule_bcet, new_task_set):
    """
    Extract offset and jitter parameters from the Gunzel schedule that match our paper
    Args:
        schedule_wcet: WCET (worst-case execution time) schedule
        task_set: task set
        schedule_bcet: BCET (best-case execution time) schedule
        new_task_set: new task set
    Returns:
        tuple: (read jitter list, write jitter list, read offset list, write offset list)
    """
    selected_read_offsets = []
    selected_write_offsets = []
    read_jitters = []
    write_jitters = []

    # From the same schedule produced by Gunzel IC, 
    # for each task , computed the events and min/max task latencies of Eq. (6), 
    # as described in Section II.B.
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



def compare_our_passive_active(jitters, num_chains, num_repeats, random_seed, periods):
    """
    Compares the "Passive Analysis" and "Active Analysis" (adjustment) experiments from the RTSS 2025 paper.

    Args:
        jitters: List of jitter values
        num_chains: List of number of task chains
        num_repeats: Number of repeats
        random_seed: Random seed
        periods: List of periods

    Returns:
        tuple: A tuple containing all results (passive result, passive failed result, passive final result, active result, active failed result, active final result)
    """
    TOLERANCE = 1e-9
    # preparing list for storing result
    results = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    final = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    false_results = {num_tasks: {per_jitter: 0 for per_jitter in jitters} for num_tasks in num_chains}
    
    results_active = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    final_active = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    false_results_active = {num_tasks: {per_jitter: 0 for per_jitter in jitters} for num_tasks in num_chains}

    for i in range(num_repeats):
        random.seed(random_seed)

        for num_tasks in num_chains:
            # Random generation matches our periods and read/write offsets
            selected_periods, selected_read_offsets, selected_write_offsets = generate_periods_and_offsets(num_tasks, periods)

            for per_jitter in jitters:
                # Passive analysis
                print(f"=========For evaluation passive our========= num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                final_e2e_max, max_reaction_time,  final_r, final_w, tasks= run_analysis_passive_our(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
                
                if final_e2e_max != 0:
                    # Sec. VI.
                    # R = DFFbase/DFFbound
                    r = max_reaction_time / final_e2e_max
                    if r > 1 + TOLERANCE:  
                        exceed = "exceed"
                    else:
                        exceed = "safe"
                else:
                    # Returns 0 if the algorithm fails
                    r = None
                    exceed = None
                    false_results[num_tasks][per_jitter] += 1  

                results[num_tasks][per_jitter].append((final_e2e_max, max_reaction_time,r,tasks,random_seed,exceed))
                final[num_tasks][per_jitter].append((final_r, final_w))

                # Active analysis
                print(f"=========For evaluation active our========= num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                final_e2e_max_active, max_reaction_time_active,  final_r_active, final_w_active, tasks_active, adjusted, inserted = run_analysis_active_our(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)

                if final_e2e_max_active != 0:
                    r_active = max_reaction_time_active / final_e2e_max_active
                    if r_active > 1 + TOLERANCE:  
                        exceed_active = "exceed"
                    else:
                        exceed_active = "safe"
                else:
                    r_active = None
                    exceed_active = None
                    false_results_active[num_tasks][per_jitter] += 1  # algorithm failed

                results_active[num_tasks][per_jitter].append((final_e2e_max_active, max_reaction_time_active, r_active, tasks_active, random_seed, exceed_active, adjusted, inserted))
                final_active[num_tasks][per_jitter].append((final_r_active, final_w_active))

        random_seed += 1

    for num_tasks in num_chains:
        for per_jitter in jitters:
            false_percentage = (false_results[num_tasks][per_jitter] / num_repeats)
            false_results[num_tasks][per_jitter] = false_percentage

            false_percentage_active = (false_results_active[num_tasks][per_jitter] / num_repeats)
            false_results_active[num_tasks][per_jitter] = false_percentage_active

    return results, false_results, final, results_active, false_results_active, final_active



def compare_Gunzel_IC(num_chains, num_repeats, random_seed, periods):
    """
    Compares our (passive, active) and paper [20] implicit communication (Gunzel IC) experiments.
    [20] M. Günzel, K.-H. Chen, N. Ueter, G. von der Brüggen, M. Dürr, and J.-J. Chen, “Timing analysis of asynchronized distributed cause- effect chains,” in Real Time and Embedded Technology and Applications Symposium (RTAS), 2021.

    Args:
        num_chains: List of number of task chains
        num_repeats: Number of repeats
        random_seed: Random seed
        periods: List of periods

    Returns:
        tuple: A tuple containing all results (passive result, passive failed result, passive final result, active result, active failed result, active final result)
    """
    TOLERANCE = 1e-9
    # preparing list for storing result
    results = {num_tasks: [] for num_tasks in num_chains}
    final = {num_tasks: [] for num_tasks in num_chains}
    false_results = {num_tasks:  0  for num_tasks in num_chains}
    
    results_active = {num_tasks:  [] for num_tasks in num_chains}
    final_active = {num_tasks:  []  for num_tasks in num_chains}
    false_results_active = {num_tasks:  0  for num_tasks in num_chains}

    for i in range(num_repeats):
        random.seed(random_seed)

        for num_tasks in num_chains:
            # Generate Gunzel task and calculate IC maximum reaction time
            ic, selected_periods, schedule_wcet, task_set, schedule_bcet, new_task_set,run_time_G = run_analysis_Gunzel_IC(num_tasks, periods)
            read_jitters, write_jitters,  selected_read_offsets, selected_write_offsets = convert_to_our_offsets(schedule_wcet, task_set, schedule_bcet, new_task_set)

            # Compare passive analysis between our and Gunzel IC
            print(f"=========For evaluation passive Gunzel IC========= num_tasks {num_tasks} Repeat {i} random_seed {random_seed} ==================")
            t_our_0 = time.perf_counter()
            # Analyzing the Gunzel tasks using our passive analysis (RTSS'2025 Alg.2)
            final_e2e_max, final_r, final_w, tasks = run_analysis_passive_Gunzel_IC(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, read_jitters, write_jitters )
            t_our_1 = time.perf_counter()
            run_time_our = t_our_1 - t_our_0

            if final_e2e_max != 0:
                r = ic  / final_e2e_max
                if r > 1 + TOLERANCE:  
                    exceed = "exceed"
                else:
                    exceed = "safe"
            else:
                r = None
                exceed = None
                false_results[num_tasks] += 1  # algorithm failed

            results[num_tasks].append((final_e2e_max, ic,r,tasks,random_seed,exceed,run_time_our,run_time_G))
            final[num_tasks].append((final_r, final_w))

            # Compare active analysis between our and Gunzel IC
            print(f"=========For evaluation active Gunzel IC========= num_tasks {num_tasks} Repeat {i} random_seed {random_seed} ==================")
            t_our_active_0 = time.perf_counter()
            # Analyzing the Gunzel tasks using our active analysis
            final_e2e_max_active, final_r_active, final_w_active, tasks_active, adjusted, inserted = run_analysis_active_Gunzel_IC(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, read_jitters, write_jitters )
            t_our_active_1 = time.perf_counter()
            run_time_our_active = t_our_active_1 - t_our_active_0

            if final_e2e_max_active != 0:
                r_active = ic / final_e2e_max_active
                if r_active > 1 + TOLERANCE:  
                    exceed_active = "exceed"
                else:
                    exceed_active = "safe"
            else:
                r_active = None
                exceed_active = None
                false_results_active[num_tasks] += 1  # algorithm failed

            results_active[num_tasks].append((final_e2e_max_active, ic, r_active, tasks_active, random_seed, exceed_active, adjusted, inserted,run_time_our_active,run_time_G))
            final_active[num_tasks].append((final_r_active, final_w_active))

        random_seed += 1

    for num_tasks in num_chains:

        false_percentage = (false_results[num_tasks] / num_repeats)
        false_results[num_tasks]= false_percentage

        false_percentage_active = (false_results_active[num_tasks] / num_repeats)
        false_results_active[num_tasks] = false_percentage_active

    return results, false_results, final, results_active, false_results_active, final_active



def compare_Gunzel_LET(jitters, num_chains, num_repeats, random_seed, periods):
    """
    Compares our (passive, active) and paper [20] LET communication (Gunzel LET) experiments.

    Args:
        jitters: zero(LET)
        num_chains: List of number of task chains
        num_repeats: Number of repeats
        random_seed: Random seed
        periods: List of periods

    Returns:
        tuple: A tuple containing all results (passive result, passive failed result, passive final result, active result, active failed result, active final result)
    """
    TOLERANCE = 1e-9
    # preparing list for storing result
    results = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    final = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    false_results = {num_tasks: {per_jitter: 0 for per_jitter in jitters} for num_tasks in num_chains}
    
    results_active = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    final_active = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    false_results_active = {num_tasks: {per_jitter: 0 for per_jitter in jitters} for num_tasks in num_chains}
    
    for i in range(num_repeats):
        random.seed(random_seed)

        for num_tasks in num_chains:
            # Generate our LET tasks' offsets...
            selected_periods, selected_read_offsets, selected_write_offsets = generate_LET(num_tasks, periods)

            for per_jitter in jitters:
                # Compare passive analysis between our and Gunzel LET
                print(f"=========For evaluation passive Gunzel LET========= num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                t_our_0 = time.perf_counter()
                # Analyzing our LET tasks using our passive analysis (RTSS'2025 Alg.2)
                final_e2e_max, final_r, final_w, tasks = run_analysis_passive_Gunzel_LET(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
                t_our_1 = time.perf_counter()
                run_time_our = t_our_1 - t_our_0

                t_G_0 = time.perf_counter()
                # Analyzing our LET tasks using Gunzel LET
                let = run_analysis_Gunzel_LET(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
                t_G_1 = time.perf_counter()
                run_time_G = t_G_1 - t_G_0

                if final_e2e_max != 0:
                    r = let / final_e2e_max
                    if r > 1 + TOLERANCE:  
                        exceed = "exceed"
                    else:
                        exceed = "safe"
                else:
                    r = None
                    exceed = None
                    false_results[num_tasks][per_jitter] += 1  # algorithm failed

                results[num_tasks][per_jitter].append((final_e2e_max, let,r,tasks,random_seed,exceed,run_time_our,run_time_G))
                final[num_tasks][per_jitter].append((final_r, final_w))

                # Compare active analysis between our and Gunzel LET
                print(f"=========For evaluation active Gunzel LET========= num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                t_our_active_0 = time.perf_counter()
                # Analyzing our LET tasks using our active analysis 
                final_e2e_max_active, final_r_active, final_w_active, tasks_active, adjusted, inserted = run_analysis_active_Gunzel_LET(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
                t_our_active_1 = time.perf_counter()
                run_time_our_active = t_our_active_1 - t_our_active_0

                if final_e2e_max_active != 0:
                    r_active = let / final_e2e_max_active
                    if r_active > 1 + TOLERANCE:  
                        exceed_active = "exceed"
                    else:
                        exceed_active = "safe"
                else:
                    r_active = None
                    exceed_active = None
                    false_results_active[num_tasks][per_jitter] += 1  # algorithm failed

                results_active[num_tasks][per_jitter].append((final_e2e_max_active, let, r_active, tasks_active, random_seed, exceed_active, adjusted, inserted, run_time_G,run_time_our_active))
                final_active[num_tasks][per_jitter].append((final_r_active, final_w_active))

        random_seed += 1

    for num_tasks in num_chains:
        for per_jitter in jitters:
            false_percentage = (false_results[num_tasks][per_jitter] / num_repeats)
            false_results[num_tasks][per_jitter] = false_percentage

            false_percentage_active = (false_results_active[num_tasks][per_jitter] / num_repeats)
            false_results_active[num_tasks][per_jitter] = false_percentage_active

    return results, false_results, final, results_active, false_results_active, final_active



def append_to_common_csv(csv_file, common_csv_file):
    """
    Append the tmp.csv file generated from a single experiment to the common_csv_file table.
    """
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
    """
    Comparison charts for our passive and active experiments
        1) Failure rate vs. number of tasks/jitter (compare_percent)
        2) Ratio (our/Gunzel) histogram (compare_histogram)
    """
    folder_name = f"{num_repeats}_{random_seed}_{timestamp}"
    folder_path = os.path.join("compare/", folder_name)

    os.makedirs(folder_path, exist_ok=True)
    
    compare_percent_plot_name = os.path.join(folder_path,  f"compare_percent_{num_repeats}_{random_seed}_{timestamp}.png")
    compare_histogram_plot_name = os.path.join(folder_path, f"compare_R_{num_repeats}_{random_seed}_{timestamp}.png")

    compare_false_percent_our(csv_files, compare_percent_plot_name)
    compare_plot_histogram_our(csv_files, compare_histogram_plot_name)
    
    print(f"Compare percent plots generated and saved to {compare_percent_plot_name} and {compare_histogram_plot_name}")



def run_Gunzel_IC(random_seed, num_repeats, common_csv_passive, common_csv_active):
    """
    Implicit communication comparison experiment (passive/active/Gunzel IC)
    RTSS'2025 fig.11, 13
    """
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000]  # periods
    num_chains = [3,5,8,10] 
    timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime("%Y%m%d_%H%M%S")

    results, false_results, final, results_active, false_results_active, final_active = compare_Gunzel_IC(num_chains, num_repeats, random_seed, periods)  

    csv_file = output_passive_Gunzel_IC(num_repeats, random_seed, timestamp, results, false_results, num_chains)
    csv_file_active = output_active_Gunzel_IC(num_repeats, random_seed, timestamp, results_active, false_results_active, num_chains)
    append_to_common_csv(csv_file, common_csv_passive)
    append_to_common_csv(csv_file_active, common_csv_active)



def run_Gunzel_LET(random_seed, num_repeats, common_csv_passive, common_csv_active):    
    """
    LET communication comparison experiment (passive-jitter=0/active-jitter=0/Gunzel LET)
    """
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000]  # periods
    num_chains = [3,5,8,10] 
    jitters = [0] # for LET
    timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime("%Y%m%d_%H%M%S")

    results, false_results, final, results_active, false_results_active, final_active = compare_Gunzel_LET(jitters, num_chains, num_repeats, random_seed, periods)

    csv_file = output_passive_Gunzel_LET(num_repeats, random_seed, timestamp, results, false_results, num_chains,jitters)
    csv_file_active = output_active_Gunzel_LET(num_repeats, random_seed, timestamp, results_active, false_results_active, num_chains,jitters)

    append_to_common_csv(csv_file, common_csv_passive)
    append_to_common_csv(csv_file_active, common_csv_active)



def run_our_passive_active(random_seed, num_repeats, common_csv_passive, common_csv_active):
    """
    LET(jitter=0)/Implicit communication(IC) comparison experiment (passive/active)
    RTSS'2025 fig.10, 12
    """
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000] 
    jitters = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5] 
    num_chains = [3, 5, 8, 10] 
    
    timestamp = datetime.datetime.fromtimestamp(int(time.time())).strftime("%Y%m%d_%H%M%S")
        
    results, false_results, final, results_active, false_results_active, final_active = compare_our_passive_active(
        jitters, num_chains, num_repeats, random_seed, periods)

    csv_file = output_passive_our(num_repeats, random_seed, timestamp, results, false_results, num_chains, jitters)
    csv_file_active = output_active_our(num_repeats, random_seed, timestamp, results_active, false_results_active, num_chains, jitters)
    append_to_common_csv(csv_file, common_csv_passive)
    append_to_common_csv(csv_file_active, common_csv_active)


def main():
    parser = argparse.ArgumentParser(description='compare our_passive, our_active, GunzelIC, GunzelLET')
    parser.add_argument('random_seed', type=int, help='random seed for the experiment')
    parser.add_argument('num_repeats', type=int, help='number of repeats for the experiment')
    parser.add_argument('--common_csv_passive', type=str, default='common_results_passive.csv', 
                        help='passive result csv file (common_results_passive.csv)')
    parser.add_argument('--common_csv_active', type=str, default='common_results_active.csv', 
                        help='active result csv file (common_results_active.csv)')
    parser.add_argument('--suffix', default='', help='different name like _IC/_LET (RTSS)')
    parser.add_argument('--alg', choices=['IC', 'LET', 'RTSS'], default='RTSS',
                    help='which algorithm to run')
    args = parser.parse_args()
    
    alg_map = {
        'IC':  run_Gunzel_IC,
        'LET':  run_Gunzel_LET,
        'RTSS': run_our_passive_active,
    }
    
    random_seed = args.random_seed
    num_repeats = args.num_repeats
    common_csv_passive   = args.common_csv_passive
    common_csv_active = args.common_csv_active

    alg = args.alg
    alg_map[alg](random_seed, num_repeats, common_csv_passive, common_csv_active)

if __name__ == "__main__":
    main()