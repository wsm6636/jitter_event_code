#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 05 10:25:52 2025

It implements the methods described in the paper
    Shumo Wang, Enrico Bini, Martina Maggio, Qingxu Deng
    "Jitter Propagation in Task Chains"

@author: Shumo Wang
"""
import argparse
import os
from plot import compare_false_percent_our
from plot import compare_plot_histogram_our
from plot import plot_R_histogram_LET
from plot import plot_R_histogram_IC
from plot import plot_runtime
import pandas as pd
import numpy as np

def suffixed(path, suffix):
    """
    Add a suffix to the file name without changing the file extension.
    Example:
        suffixed("result.png", "_IC") -> "result_IC.png"
    """
    if not suffix:
        return path
    base, ext = os.path.splitext(path)
    return f"{base}{suffix}{ext}"


def sort_csv_by_policy(csv_file, suffix=''):
    """
    Sorting strategy:
    - RTSS (suffix == '' or suffix == '_RTSS'): Sort by num_tasks first, then by per_jitter
    - Other (IC/LET): Sort by num_tasks only
    After sorting, directly replace the original file.
    """
    df = pd.read_csv(csv_file)
    is_rtss = (suffix == '') or (suffix == '_RTSS')
    if is_rtss and {'num_tasks', 'per_jitter'}.issubset(df.columns):
        df = df.sort_values(by=['num_tasks', 'per_jitter'], ascending=[True, True])
    elif 'num_tasks' in df.columns:
        df = df.sort_values(by=['num_tasks'], ascending=True)
    df.to_csv(csv_file, index=False)
    print(f"[sort] {csv_file} suffix={suffix or 'None'} ")


def add_final_percent_column_safe(csv_file, out_file):
    """
    Add a column named finalpercent to the CSV file:
    For each group (num_tasks, per_jitter for our result) or just (num_tasks for IC/LET), calculate the proportion of final_e2e_max==0.
    """
    df = pd.read_csv(csv_file)

    if 'per_jitter' in df.columns:
        # our experiments
        ratio = (df.groupby(['num_tasks', 'per_jitter'])['final_e2e_max']
                    .apply(lambda x: (x == 0).mean())
                    .reset_index(name='finalpercent'))
        df = df.merge(ratio, on=['num_tasks', 'per_jitter'], how='left')
    else:
        # LET/IC experiments without per_jitter column
        ratio = (df.groupby(['num_tasks'])['final_e2e_max']
                    .apply(lambda x: (x == 0).mean())
                    .reset_index(name='finalpercent'))
        df = df.merge(ratio, on=['num_tasks'], how='left')

    df['finalpercent'] = df['finalpercent'].astype(float)
    df.to_csv(out_file, index=False)
    return out_file
    


def filter_and_export_csv_passive(csv_file_path, num_chains, data_output_dir=None,suffix=''):
    """
    Filters passive results by task chain length and exports sub-CSV files. Used for plotting figures in latex for papers.
    Returns:
        all_data_file : List of complete data files for each num_tasks task
        jitter_20_files: List of data files for only the 20% jitter (returns empty if per_jitter is not specified)
    """
    if data_output_dir is None:
        parent_dir = os.path.dirname(csv_file_path)
        data_dir = os.path.join(parent_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: File {csv_file_path} not found.")
        return [], []
    
    all_data_files = []
    jitter_20_files = []
    
    has_jitter = 'per_jitter' in df.columns 

    for num_tasks in num_chains:
        # Filter all data for current num_tasks
        all_data = df[df['num_tasks'] == num_tasks]
        # Define file paths
        all_data_file = suffixed(os.path.join(data_output_dir, f"passive_data{num_tasks}.csv"), suffix)
        
        # Save all data for current num_tasks
        if not all_data.empty:
            all_data.to_csv(all_data_file, index=False)
            all_data_files.append(all_data_file)
            print(f"All data for {num_tasks} tasks saved to {all_data_file} ({len(all_data)} rows)")
        else:
            print(f"Warning: No data found for {num_tasks} tasks")

        if has_jitter:
            # Filter data for current num_tasks with per_jitter = 20%
            # LET/IC experiments without per_jitter column
            jitter_20_data = df[(df['num_tasks'] == num_tasks) & (df['per_jitter'] == 0.2)]
            jitter_20_file = suffixed(os.path.join(data_output_dir, f"passive_data{num_tasks}_20per.csv"), suffix)
    
            # Save 20% jitter data for current num_tasks
            if not jitter_20_data.empty:
                jitter_20_data.to_csv(jitter_20_file, index=False)
                jitter_20_files.append(jitter_20_file)
                print(f"20% jitter data for {num_tasks} tasks saved to {jitter_20_file} ({len(jitter_20_data)} rows)")
            else:
                print(f"Warning: No 20% jitter data found for {num_tasks} tasks")
            
    if has_jitter:
        return all_data_files, jitter_20_files
    
    return all_data_files



def filter_and_export_csv_active(csv_file_path, num_chains, data_output_dir=None,suffix=''):
    """
    Filters active results by task chain length and exports sub-CSV files. Used for plotting figures in latex for papers.
    Returns:
        all_data_file : List of complete data files for each num_tasks task
        jitter_20_files: List of data files for only the 20% jitter (returns empty if per_jitter is not specified)
    """
    if data_output_dir is None:
        parent_dir = os.path.dirname(csv_file_path)
        data_dir = os.path.join(parent_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: File {csv_file_path} not found.")
        return [], []
    
    all_data_files = []
    jitter_20_files = []
    has_jitter = 'per_jitter' in df.columns
    # Process each num_tasks value
    for num_tasks in num_chains:
        # Filter all data for current num_tasks
        all_data = df[df['num_tasks'] == num_tasks]
        # Define file paths
        all_data_file = suffixed(os.path.join(data_output_dir, f"active_data{num_tasks}.csv"), suffix)
        # Save all data for current num_tasks
        if not all_data.empty:
            all_data.to_csv(all_data_file, index=False)
            all_data_files.append(all_data_file)
            print(f"All data for {num_tasks} tasks saved to {all_data_file}")
        else:
            print(f"Warning: No data found for {num_tasks} tasks")
        
        if has_jitter:
            # LET/IC experiments without per_jitter column
            # Filter data for current num_tasks with per_jitter = 20%
            jitter_20_data = df[(df['num_tasks'] == num_tasks) & (df['per_jitter'] == 0.2)]
            jitter_20_file = suffixed(os.path.join(data_output_dir, f"active_data{num_tasks}_20per.csv"), suffix)
            # Save 20% jitter data for current num_tasks
            if not jitter_20_data.empty:
                jitter_20_data.to_csv(jitter_20_file, index=False)
                jitter_20_files.append(jitter_20_file)
                print(f"20% jitter data for {num_tasks} tasks saved to {jitter_20_file}")
            else:
                print(f"Warning: No 20% jitter data found for {num_tasks} tasks")

    if has_jitter:
        return all_data_files, jitter_20_files
    return all_data_files


def calculate_boxplot_stats(data):
    """Calculating box plot statistics"""
    sorted_data = np.sort(data.dropna())   
    if len(sorted_data) == 0:
        return None
        
    q1 = np.percentile(sorted_data, 25)
    median = np.percentile(sorted_data, 50)
    q3 = np.percentile(sorted_data, 75)
    iqr = q3 - q1
    
    lower_whisker = max(sorted_data.min(), q1 - 1.5 * iqr)
    upper_whisker = min(sorted_data.max(), q3 + 1.5 * iqr)
    
    return {
        'median': median,
        'q1': q1,
        'q3': q3,
        'lower_whisker': lower_whisker,
        'upper_whisker': upper_whisker,
        'min': sorted_data.min(),
        'max': sorted_data.max()
    }


def process_csv_file(input_file, output_file=None):
    """
    Process the CSV file and calculate the average runtime and boxplot statistics.
    Retain the necessary columns ('num_tasks', 'run_time_our_avg', 'run_time_G_avg')
    And add a column for the boxplot statistics.
    """
    try:
        df = pd.read_csv(input_file)
        
        required = {'num_tasks', 'run_time_our', 'run_time_G'}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            print(f"error: {input_file} miss {list(missing)}")
            return False
        
        grouped = df.groupby('num_tasks')
        
        result_rows = []
        
        for num_tasks, group in grouped:
            # avg
            avg_our = group['run_time_our'].mean()
            avg_g = group['run_time_G'].mean()
            
            # boxplot
            our_stats = calculate_boxplot_stats(group['run_time_our'])
            g_stats = calculate_boxplot_stats(group['run_time_G'])
            
            if our_stats is None or g_stats is None:
                continue
            
            row = {
                'num_tasks': num_tasks,
                'run_time_our_avg': avg_our,
                'run_time_G_avg': avg_g,
                # Our Method
                'run_time_our_median': our_stats['median'],
                'run_time_our_q1': our_stats['q1'],
                'run_time_our_q3': our_stats['q3'],
                'run_time_our_lower_whisker': our_stats['lower_whisker'],
                'run_time_our_upper_whisker': our_stats['upper_whisker'],
                'run_time_our_min': our_stats['min'],
                'run_time_our_max': our_stats['max'],
                # G Method
                'run_time_G_median': g_stats['median'],
                'run_time_G_q1': g_stats['q1'],
                'run_time_G_q3': g_stats['q3'],
                'run_time_G_lower_whisker': g_stats['lower_whisker'],
                'run_time_G_upper_whisker': g_stats['upper_whisker'],
                'run_time_G_min': g_stats['min'],
                'run_time_G_max': g_stats['max'],

                'data_count': len(group)
            }
            
            result_rows.append(row)
        
        result_df = pd.DataFrame(result_rows)
        
        if output_file is None:
            base, ext = os.path.splitext(input_file)
            output_file = f"{base}_processed{ext}"
        
        result_df.to_csv(output_file, index=False)
        print(f"save: {output_file}")

        for _, row in result_df.iterrows():
            print(f"  num_tasks={int(row['num_tasks'])}: "
                    f"Our_avg={row['run_time_our_avg']:.4f}s, "
                    f"G_avg={row['run_time_G_avg']:.4f}s, "
                    f"data count={int(row['data_count'])}")
        
        return True
        
    except FileNotFoundError:
        print(f"error: can not find {input_file}")
        return False
    except pd.errors.EmptyDataError:
        print(f"error: {input_file} empty")
        return False
    except Exception as e:
        print(f"error {input_file}: {str(e)}")
        return False


def generate_final_comparison(common_csv_passive, common_csv_active, suffix=''):
    """
    Main Process:
        1. Check file existence;
        2. Add finalpercent;
        3. Draw corresponding group graphs based on suffix (_IC / _LET / _RTSS);
        4. Split and export sub-csv files.
    """
    if not os.path.exists(common_csv_passive):
        print(f"error: can not found {common_csv_passive}")
        return False
        
    if not os.path.exists(common_csv_active):
        print(f"error: can not found {common_csv_active}")
        return False
    
    # use the path of common_csv_passive
    output_dir = os.path.dirname(os.path.abspath(common_csv_passive))
    data_output_dir = f"{output_dir}/data"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_output_dir, exist_ok=True)

    # Get the final complete CSV file
    common_csv_passive = add_final_percent_column_safe(common_csv_passive, common_csv_passive)
    common_csv_active = add_final_percent_column_safe(common_csv_active, common_csv_active)
    
    sort_csv_by_policy(common_csv_passive, suffix)
    sort_csv_by_policy(common_csv_active, suffix)

    # Drawing different file types
    csv_files = [common_csv_passive, common_csv_active]
    compare_percent_plot_our  = suffixed(os.path.join(output_dir, "final_compare_percent_our.png"), suffix)
    compare_histogram_plot_our = suffixed(os.path.join(output_dir, "final_compare_histogram_our.png"), suffix)
    runtime_plot_passive  = suffixed(os.path.join(output_dir, "final_runtime_passive.png"), suffix)
    histogram_plot_passive = suffixed(os.path.join(output_dir, "final_R_histogram_passive.png"), suffix)
    runtime_plot_active  = suffixed(os.path.join(output_dir, "final_runtime_active.png"), suffix)
    histogram_plot_active = suffixed(os.path.join(output_dir, "final_R_histogram_active.png"), suffix)

    if suffix == '_IC':
        plot_runtime(common_csv_passive, runtime_plot_passive,tag='passive')
        plot_R_histogram_IC(common_csv_passive, histogram_plot_passive,tag='passive')
        process_csv_file(common_csv_passive)
        plot_runtime(common_csv_active, runtime_plot_active,tag='active')
        plot_R_histogram_IC(common_csv_active, histogram_plot_active,tag='active')
        process_csv_file(common_csv_active)
    elif suffix == '_LET':
        plot_runtime(common_csv_passive, runtime_plot_passive,tag='passive')
        plot_R_histogram_LET(common_csv_passive, histogram_plot_passive,tag='passive')
        process_csv_file(common_csv_passive)
        plot_runtime(common_csv_active, runtime_plot_active,tag='active')
        plot_R_histogram_LET(common_csv_active, histogram_plot_active,tag='active')
        process_csv_file(common_csv_active)
    else:
        compare_false_percent_our(csv_files, compare_percent_plot_our)
        compare_plot_histogram_our(csv_files, compare_histogram_plot_our)

    # Split files into different types
    filter_and_export_csv_passive(common_csv_passive, [3, 5, 8, 10], data_output_dir, suffix)
    filter_and_export_csv_active(common_csv_active, [3, 5, 8, 10], data_output_dir, suffix)
    print(f"Filtered CSV files saved in {data_output_dir}")


def main():
    parser = argparse.ArgumentParser(description='generate final comparison plots from CSV files')
    parser.add_argument('--common_csv_passive', type=str, default='common_results_passive.csv',
                        help='passive result csv file (common_results_passive.csv)')
    parser.add_argument('--common_csv_active', type=str, default='common_results_active.csv',
                        help='active result csv file (common_results_active.csv)')
    parser.add_argument('--suffix', default='', help='filename suffix like _IC / _LET')
    
    args = parser.parse_args()
    if os.path.exists(args.common_csv_passive):
        with open(args.common_csv_passive, 'r') as f:
            lines = len(f.readlines()) - 1  
        print(f"rtss result total rows: {lines}")
    else:
        print(f"can not find {args.common_csv_passive}")
    
    if os.path.exists(args.common_csv_active):
        with open(args.common_csv_active, 'r') as f:
            lines = len(f.readlines()) - 1  
        print(f"adjust result total rows: {lines}")
    else:
        print(f"can not find {args.common_csv_active}")
    
    generate_final_comparison(args.common_csv_passive, args.common_csv_active, suffix=args.suffix)


if __name__ == "__main__":
    exit(main())
