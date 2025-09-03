#!/usr/bin/env python3

import argparse
import os
import datetime
import time
from plot import compare_line_chart_from_csv
from plot import compare_plot_histogram
from plot import plot_histogram_LET
from plot import plot_histogram_MRT
from plot import plot_runtime

import pandas as pd


def suffixed(path, suffix):
    if not suffix:
        return path
    base, ext = os.path.splitext(path)
    return f"{base}{suffix}{ext}"

def add_final_percent_column_safe(csv_file, out_file):
    df = pd.read_csv(csv_file)
   
    # Check if the required columns exist and handle different experiment types
    if 'per_jitter' in df.columns:
        # Original jitter-based experiments
        ratio = (df.groupby(['num_tasks', 'per_jitter'])['final_e2e_max']
                   .apply(lambda x: (x == 0).mean())
                   .reset_index(name='finalpercent'))
        df = df.merge(ratio, on=['num_tasks', 'per_jitter'], how='left')
    else:
        # LET/MRT experiments without per_jitter column
        ratio = (df.groupby(['num_tasks'])['final_e2e_max']
                   .apply(lambda x: (x == 0).mean())
                   .reset_index(name='finalpercent'))
        df = df.merge(ratio, on=['num_tasks'], how='left')

    df['finalpercent'] = df['finalpercent'].astype(float)
    df.to_csv(out_file, index=False)
    return out_file
    


def filter_and_export_csv(csv_file_path, num_chains, data_output_dir=None,suffix=''):
    if data_output_dir is None:
            # Get parent directory of the CSV file and create data subdirectory
        parent_dir = os.path.dirname(csv_file_path)
        data_dir = os.path.join(parent_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

    # Read the original CSV file
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
        all_data_file = suffixed(os.path.join(data_output_dir, f"data{num_tasks}.csv"), suffix)
        
        # Save all data for current num_tasks
        if not all_data.empty:
            all_data.to_csv(all_data_file, index=False)
            all_data_files.append(all_data_file)
            print(f"All data for {num_tasks} tasks saved to {all_data_file} ({len(all_data)} rows)")
        else:
            print(f"Warning: No data found for {num_tasks} tasks")

        if has_jitter:
            # Filter data for current num_tasks with per_jitter = 20%
            jitter_20_data = df[(df['num_tasks'] == num_tasks) & (df['per_jitter'] == 0.2)]
            jitter_20_file = suffixed(os.path.join(data_output_dir, f"data{num_tasks}_20per.csv"), suffix)
    
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


def filter_and_export_csv_adjust(csv_file_path, num_chains, data_output_dir=None,suffix=''):
    if data_output_dir is None:
    # Get parent directory of the CSV file and create data subdirectory
        parent_dir = os.path.dirname(csv_file_path)
        data_dir = os.path.join(parent_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
    
    # Read the original CSV file
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
        all_data_file = suffixed(os.path.join(data_output_dir, f"adjust_data{num_tasks}.csv"), suffix)
        # Save all data for current num_tasks
        if not all_data.empty:
            all_data.to_csv(all_data_file, index=False)
            all_data_files.append(all_data_file)
            print(f"All data for {num_tasks} tasks saved to {all_data_file}")
        else:
            print(f"Warning: No data found for {num_tasks} tasks")
        
        if has_jitter:
        # Filter data for current num_tasks with per_jitter = 20%
            jitter_20_data = df[(df['num_tasks'] == num_tasks) & (df['per_jitter'] == 0.2)]
            jitter_20_file = suffixed(os.path.join(data_output_dir, f"adjust_data{num_tasks}_20per.csv"), suffix)
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



def generate_final_comparison(common_csv, common_csv_adjust, suffix=''):

    if not os.path.exists(common_csv):
        print(f"error: can not found {common_csv}")
        return False
        
    if not os.path.exists(common_csv_adjust):
        print(f"error: can not found {common_csv_adjust}")
        return False
    

    output_dir = os.path.dirname(os.path.abspath(common_csv))
    data_output_dir = f"{output_dir}/data"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_output_dir, exist_ok=True)

    common_csv = add_final_percent_column_safe(common_csv, common_csv)
    common_csv_adjust = add_final_percent_column_safe(common_csv_adjust, common_csv_adjust)
    
    csv_files = [common_csv, common_csv_adjust]
    compare_percent_plot  = suffixed(os.path.join(output_dir, "final_compare_percent.png"), suffix)
    compare_histogram_plot = suffixed(os.path.join(output_dir, "final_compare_histogram.png"), suffix)
    runtime_plot  = suffixed(os.path.join(output_dir, "final_runtime.png"), suffix)
    histogram_plot = suffixed(os.path.join(output_dir, "final_R_histogram.png"), suffix)
    runtime_plot_adjust  = suffixed(os.path.join(output_dir, "final_runtime_adjust.png"), suffix)
    histogram_plot_adjust = suffixed(os.path.join(output_dir, "final_R_histogram_adjust.png"), suffix)

    if suffix == '_MRT':
        plot_runtime(common_csv, runtime_plot)
        plot_histogram_MRT(common_csv, histogram_plot)
        plot_runtime(common_csv_adjust, runtime_plot_adjust)
        plot_histogram_MRT(common_csv_adjust, histogram_plot_adjust)
    elif suffix == '_LET':
        plot_runtime(common_csv, runtime_plot)
        plot_histogram_LET(common_csv, histogram_plot)
        plot_runtime(common_csv_adjust, runtime_plot_adjust)
        plot_histogram_LET(common_csv_adjust, histogram_plot_adjust)
    else:
        compare_line_chart_from_csv(csv_files, compare_percent_plot)
        compare_plot_histogram(csv_files, compare_histogram_plot)

    filter_and_export_csv(common_csv, [3, 5, 8, 10], data_output_dir)
    filter_and_export_csv_adjust(common_csv_adjust, [3, 5, 8, 10], data_output_dir)
    print(f"Filtered CSV files saved in {data_output_dir}")


def main():
    # parser = argparse.ArgumentParser(description='generate final comparison plots from CSV files')
    # parser.add_argument('--common_csv', type=str, default='common_results.csv',
    #                     help='rtss result csv file (common_results.csv)')
    # parser.add_argument('--common_csv_adjust', type=str, default='common_results_adjust.csv',
    #                     help='adjust result csv file (common_results_adjust.csv)')
    # parser.add_argument('--suffix', default='', help='filename suffix like _MRT / _LET')
    # 
    
    # if os.path.exists(args.common_csv):
    #     with open(args.common_csv, 'r') as f:
    #         lines = len(f.readlines()) - 1  
    #     print(f"rtss result total rows: {lines}")
    # else:
    #     print(f"can not find {args.common_csv}")
    
    # if os.path.exists(args.common_csv_adjust):
    #     with open(args.common_csv_adjust, 'r') as f:
    #         lines = len(f.readlines()) - 1  
    #     print(f"adjust result total rows: {lines}")
    # else:
    #     print(f"can not find {args.common_csv_adjust}")
    
    # generate_final_comparison(args.common_csv, args.common_csv_adjust, suffix=args.suffix)
    parser = argparse.ArgumentParser(description="Plot histograms from a CSV file.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing the data.")
    parser.add_argument("runtime_plt_name", type=str, help="Name of the output plot file for R values.")
    args = parser.parse_args()
    filter_and_export_csv_adjust(args.csv_file, [3, 5, 8, 10], args.runtime_plt_name)

if __name__ == "__main__":
    exit(main())