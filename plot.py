#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 05 10:25:52 2025

It implements the methods described in the paper
    Shumo Wang, Enrico Bini, Martina Maggio, Qingxu Deng
    "Jitter in Task Chains"

@author: Shumo Wang
"""
import os
import matplotlib.pyplot as plt
import csv
import numpy as np
import argparse 
import pandas as pd
from matplotlib.gridspec import GridSpec


def plot_R_histogram_our(csv_file,R_plot_name,tag='passive'):
    """
    Read csv_file, retain only the data with per_jitter = 20%,
    group by num_tasks, plot an R value histogram (50 bins evenly spaced from 0 to 1.05),
    and calculate the proportion of R values greater than 1.
    R = DFFbase/DFFbound
    """
    num_tasks_to_r_values = {}
    r_exceed_count = 0  # Counter for R values exceeding 1.0
    total_rows = 0 
    TOLERANCE = 1e-9
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            total_rows += 1 
            per_jitter = float(row['per_jitter'])
            r_value = float(row['R']) if row['R'] else None
            num_tasks = int(row['num_tasks'])

            if per_jitter == 0.2 and r_value is not None:  # Only consider per_jitter = 20%
                if num_tasks not in num_tasks_to_r_values:
                    num_tasks_to_r_values[num_tasks] = []
                num_tasks_to_r_values[num_tasks].append(r_value)
                if r_value > 1 + TOLERANCE:
                    print(f"Warning: R value {r_value} exceeds 1.0 for per_jitter={per_jitter}. This may indicate an error in the data.")
                    r_exceed_count += 1  

    if total_rows == 0:
        print("No data found.")
        return
                
    R_exceed_percentage = r_exceed_count / total_rows * 100 if total_rows > 0 else 0

    num_num_tasks = len(num_tasks_to_r_values)
    if num_num_tasks == 0:
        print("No valid data found for the specified conditions.")
        return

    num_columns = 2  
    num_rows = (num_num_tasks + num_columns - 1) // num_columns

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, num_num_tasks))  

    for idx, (num_tasks, r_values) in enumerate(num_tasks_to_r_values.items()):
        ax = axes[idx]
        num_bins = 50
        bin_range = (0, 1.05)
        bin_width = (bin_range[1] - bin_range[0]) / num_bins

        counts, bin_edges = np.histogram(r_values, bins=num_bins, range=bin_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, counts, width=bin_width, alpha=0.7, align='center', color=colors[idx], label=f'num_tasks={num_tasks}')

        r_values_greater_than_1 = len([r for r in r_values if r > 1])
        percentage_greater_than_1 = (r_values_greater_than_1 / len(r_values)) * 100 if len(r_values) > 0 else 0

        ax.set_title(f"num_tasks = {num_tasks} - our ({tag}) - Data Count: {len(r_values)}")
        ax.set_xlabel(f"R_exceed_percentage = {percentage_greater_than_1:.2f}%")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True)

    for idx in range(num_num_tasks, num_rows * num_columns):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.suptitle(f"Distribution of R values for different num_tasks (per_jitter=20%),R_exceed_percentage={R_exceed_percentage}", fontsize=16, y=1.05)

    plt.savefig(R_plot_name)



def plot_R_histogram_LET(csv_file,R_plot_name_LET,tag='passive'):
    """
    No filtering per_jitter required
    R = DFF_Gunzel_LET/DFFbound
    """
    num_tasks_to_r_values = {}
    r_exceed_count = 0  # Counter for R values exceeding 1.0
    total_rows = 0 
    TOLERANCE = 1e-9
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            total_rows += 1 
            r_value = float(row['R']) if row['R'] else None
            num_tasks = int(row['num_tasks'])

            if r_value is not None:  
                if num_tasks not in num_tasks_to_r_values:
                    num_tasks_to_r_values[num_tasks] = []
                num_tasks_to_r_values[num_tasks].append(r_value)
                if r_value > 1 + TOLERANCE:
                    print(f"Warning: R value {r_value} exceeds 1.0. This may indicate an error in the data.")
                    r_exceed_count += 1  

    if total_rows == 0:
        print("No data found.")
        return
                
    R_exceed_percentage = r_exceed_count / total_rows * 100 if total_rows > 0 else 0

    num_num_tasks = len(num_tasks_to_r_values)
    if num_num_tasks == 0:
        print("No valid data found for the specified conditions.")
        return

    num_columns = 2  
    num_rows = (num_num_tasks + num_columns - 1) // num_columns

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, num_num_tasks))  

    for idx, (num_tasks, r_values) in enumerate(num_tasks_to_r_values.items()):
        ax = axes[idx]
        num_bins = 50
        bin_range = (0, 1.05)
        bin_width = (bin_range[1] - bin_range[0]) / num_bins

        counts, bin_edges = np.histogram(r_values, bins=num_bins, range=bin_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, counts, width=bin_width, alpha=0.7, align='center', color=colors[idx], label=f'num_tasks={num_tasks}')

        r_values_greater_than_1 = len([r for r in r_values if r > 1])
        percentage_greater_than_1 = (r_values_greater_than_1 / len(r_values)) * 100 if len(r_values) > 0 else 0

        ax.set_title(f"num_tasks = {num_tasks} - LET ({tag}) - Data Count: {len(r_values)}")
        ax.set_xlabel(f"R_exceed_percentage = {percentage_greater_than_1:.2f}%")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True)

    for idx in range(num_num_tasks, num_rows * num_columns):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.suptitle(f"Distribution of R values for different num_tasks (LET),R_exceed_percentage={R_exceed_percentage}", fontsize=16, y=1.05)


    plt.savefig(R_plot_name_LET)



def plot_R_histogram_IC(csv_file,R_plot_name_IC,tag='passive'):
    """
    No filtering per_jitter required
    R = DFF_Gunzel_IC/DFFbound
    """
    num_tasks_to_r_values = {}
    r_exceed_count = 0  # Counter for R values exceeding 1.0
    total_rows = 0 
    TOLERANCE = 1e-9
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            total_rows += 1 
            r_value = float(row['R']) if row['R'] else None
            num_tasks = int(row['num_tasks'])

            if r_value is not None:
                if num_tasks not in num_tasks_to_r_values:
                    num_tasks_to_r_values[num_tasks] = []
                num_tasks_to_r_values[num_tasks].append(r_value)
                if r_value > 1 + TOLERANCE:
                    print(f"Warning: R value {r_value} exceeds 1.0 . This may indicate an error in the data.")
                    r_exceed_count += 1  

    if total_rows == 0:
        print("No data found.")
        return
                
    R_exceed_percentage = r_exceed_count / total_rows * 100 if total_rows > 0 else 0

    num_num_tasks = len(num_tasks_to_r_values)
    if num_num_tasks == 0:
        print("No valid data found for the specified conditions.")
        return

    num_columns = 2  
    num_rows = (num_num_tasks + num_columns - 1) // num_columns

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, num_num_tasks))  

    for idx, (num_tasks, r_values) in enumerate(num_tasks_to_r_values.items()):
        ax = axes[idx]
        num_bins = 50
        bin_range = (0, 1.05)
        bin_width = (bin_range[1] - bin_range[0]) / num_bins

        counts, bin_edges = np.histogram(r_values, bins=num_bins, range=bin_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, counts, width=bin_width, alpha=0.7, align='center', color=colors[idx], label=f'num_tasks={num_tasks}')

        r_values_greater_than_1 = len([r for r in r_values if r > 1])
        percentage_greater_than_1 = (r_values_greater_than_1 / len(r_values)) * 100 if len(r_values) > 0 else 0

        ax.set_title(f"num_tasks = {num_tasks} - IC ({tag}) - Data Count: {len(r_values)}")
        ax.set_xlabel(f"R_exceed_percentage = {percentage_greater_than_1:.2f}%")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True)

    for idx in range(num_num_tasks, num_rows * num_columns):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.suptitle(f"Distribution of R values for different num_tasks (IC),R_exceed_percentage={R_exceed_percentage}", fontsize=16, y=1.05)

    plt.savefig(R_plot_name_IC)



def plot_runtime(csv_path, runtime_name, tag='passive'):
    """
    Read csv, group by num_tasks, and calculate the average of run_time_G (Gunzel) and run_time_our (our).
    """
    df = pd.read_csv(csv_path)

    avg = (df
            .groupby('num_tasks')[['run_time_G', 'run_time_our']]
            .mean()
            .reset_index()
            .sort_values('num_tasks'))
    plt.figure(figsize=(6, 4))
    plt.plot(avg['num_tasks'], avg['run_time_G'],
                marker='o', label='run_time_G (Average)')
    plt.plot(avg['num_tasks'], avg['run_time_our'],
                marker='^', label='run_time_our (Average)')
    plt.yscale('log')
    plt.xlabel('num_tasks')
    plt.ylabel('Average Runtime (s)')
    plt.title(f' Average Runtime vs. num_tasks ({tag})')

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(runtime_name, dpi=300)

    

def plot_false_percent(csv_file, percent_plot_name, tag='passive'):
    """
    Group by num_tasks and plot a line with false_percentage varying with jitter.
    """
    jitter_to_false_percentage = {}
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            per_jitter = float(row['per_jitter'])
            false_percentage = float(row['false_percentage'])  
            num_tasks = int(row['num_tasks'])  

            if num_tasks not in jitter_to_false_percentage:
                jitter_to_false_percentage[num_tasks] = {}

            if per_jitter not in jitter_to_false_percentage[num_tasks]:
                jitter_to_false_percentage[num_tasks][per_jitter] = []

            jitter_to_false_percentage[num_tasks][per_jitter].append(false_percentage)

    plt.figure(figsize=(10, 6))
    for num_tasks, jitter_data in jitter_to_false_percentage.items():
        jitter_percent = [jitter * 100 for jitter in sorted(jitter_data.keys())]
        false_percentages = [np.mean(jitter_data[jitter]) * 100 for jitter in sorted(jitter_data.keys())]
        plt.plot(jitter_percent, false_percentages, label=f"num_tasks={num_tasks}", marker='o')

    plt.title(f"False Percentage vs. Jitter ({tag})")
    plt.xlabel("Jitter Percentage (%)")
    plt.ylabel("False Percentage (%)")
    plt.legend()
    plt.grid(True)
    plt.xticks(jitter_percent)  
    plt.savefig(f"{percent_plot_name}")



def compare_plot_histogram_our(csv_files, compare_histogram_our_name):
    """
    Comparing two experiments in our paper,
    csv_files (passive and active),
    each with per_jitter = 20% data, plotting histograms side by side by num_tasks.
    """
    dfs = [pd.read_csv(file) for file in csv_files]

    dfs = [df[df['per_jitter'] == 0.2] for df in dfs]

    num_tasks_list = sorted(set.union(*[set(df['num_tasks'].unique()) for df in dfs]))

    fig = plt.figure(figsize=(20, 10 * len(num_tasks_list)))
    outer_grid = GridSpec(len(num_tasks_list), len(csv_files), wspace=0.4, hspace=0.4)

    LABELS = ['passive', 'active']

    colors = plt.cm.tab10(np.linspace(0, 1, len(num_tasks_list)))
    TOLERANCE = 1e-9
    for idx, num_tasks in enumerate(num_tasks_list):
        for file_idx, df in enumerate(dfs):
            ax = fig.add_subplot(outer_grid[idx, file_idx])
            df_task = df[df['num_tasks'] == num_tasks]

            r_values = df_task['R'].dropna().values
            r_exceed_count = (r_values > (1 + TOLERANCE)).sum()
            R_exceed_percentage = (r_exceed_count / len(r_values)) * 100 if len(r_values) > 0 else 0

            num_bins = 50
            bin_range = (0, 1.05)
            bin_width = (bin_range[1] - bin_range[0]) / num_bins

            counts, bin_edges = np.histogram(r_values, bins=num_bins, range=bin_range)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax.bar(bin_centers, counts, width=bin_width, alpha=0.7, align='center', color=colors[idx], label=f'num_tasks={num_tasks}')

            label = LABELS[file_idx] 
            ax.set_title(f"num_tasks = {num_tasks}  (per_jitter=20%) - our({label}) - Data Count: {len(r_values)}")
            ax.set_xlabel("R_exceed_percentage = {:.2f}%".format(R_exceed_percentage))
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(True)

    plt.savefig(compare_histogram_our_name)



def compare_false_percent_our(csv_files, compare_plot_name):
    """
    Comparing the two experiments in our paper,
    csv_files (passive and active),
    draw a line graph showing the False Percentage as Jitter changes.
    """
    num_csv_files = len(csv_files)
    num_columns = 2
    num_rows = (num_csv_files + num_columns - 1) // num_columns

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows))
    axes = axes.flatten()
    LABELS = ['passive', 'active']
    for idx, csv_file in enumerate(csv_files):
        ax = axes[idx]
        try:
            df = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"File not found: {csv_file}")
            continue
        except pd.errors.EmptyDataError:
            print(f"No data in CSV file: {csv_file}")
            continue

        label = LABELS[idx] 
        
        grouped_by_num_tasks = df.groupby('num_tasks')

        for num_tasks, group in grouped_by_num_tasks:
            group_sorted = group.sort_values(by='per_jitter')

            per_jitters = group_sorted['per_jitter'] * 100
            false_percentages = group_sorted['finalpercent']

            ax.plot(per_jitters, false_percentages, label=f'num_tasks={num_tasks}', marker='o')

        ax.set_title(f"False Percentage vs. Jitter ({label})")
        ax.set_xlabel("Jitter Percentage (%)")
        ax.set_ylabel("False Percentage (%)")
        ax.legend()
        ax.grid(True)
    
    for idx in range(num_csv_files, num_rows * num_columns):
        axes[idx].axis('off')
    
    y_min = min(ax.get_ylim()[0] for ax in axes if ax.has_data())
    y_max = max(ax.get_ylim()[1] for ax in axes if ax.has_data())
    for ax in axes:
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(compare_plot_name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot histograms from a CSV file.")
    # parser.add_argument("csv_file", type=str, help="Path to the CSV file containing the data.")
    parser.add_argument("csv_files", type=str, nargs='+', help="Paths to the CSV files containing the data.")
    parser.add_argument("name", type=str)
    args = parser.parse_args()

    compare_plot_histogram_our(args.csv_files, args.name)