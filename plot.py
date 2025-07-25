import matplotlib.pyplot as plt
import csv
import numpy as np
import argparse 
from scipy.stats import gaussian_kde
import pandas as pd
import ast
import os
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn as sns

def plot_histogram_from_csv(csv_file,R_plot_name):
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

        ax.set_title(f"num_tasks = {num_tasks} (per_jitter=20%) - Data Count: {len(r_values)}")
        ax.set_xlabel(f"R_exceed_percentage = {percentage_greater_than_1:.2f}%")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True)

    for idx in range(num_num_tasks, num_rows * num_columns):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.suptitle(f"Distribution of R values for different num_tasks (per_jitter=20%),R_exceed_percentage={R_exceed_percentage}", fontsize=16, y=1.05)


    plt.savefig(R_plot_name)
    # plt.show()



def plot_line_chart_from_csv(csv_file, percent_plot_name):
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

    plt.title("False Percentage vs. Jitter for Different Number of Tasks")
    plt.xlabel("Jitter Percentage (%)")
    plt.ylabel("False Percentage (%)")
    plt.legend()
    plt.grid(True)
    plt.xticks(jitter_percent)  
    plt.savefig(f"{percent_plot_name}")
    # plt.show()


def plot_final_e2e_max(e2e_plot_name, csv_file):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)

    # 检查必要列是否存在
    required_columns = ['num_tasks', 'per_jitter', 'final_e2e_max', 'false_percentage']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("CSV 文件缺少必要列：num_tasks, per_jitter, final_e2e_max, false_percentage")

    # 获取唯一的 num_tasks 和 per_jitter
    num_tasks_values = sorted(df['num_tasks'].unique())
    per_jitter_values = sorted(df['per_jitter'].unique())

    # 创建子图
    fig, axs = plt.subplots(1, len(num_tasks_values), figsize=(18, 6), sharey=True)

    # 如果只有一个子图，确保 axs 是列表
    if len(num_tasks_values) == 1:
        axs = [axs]

    # 为每个 per_jitter 分配颜色
    colors = plt.cm.viridis(np.linspace(0, 1, len(per_jitter_values)))

    for i, num_tasks in enumerate(num_tasks_values):
        ax = axs[i]
        for j, per_jitter in enumerate(per_jitter_values):
            subset = df[(df['num_tasks'] == num_tasks) & (df['per_jitter'] == per_jitter)]
            if subset.empty:
                continue

            # 按失败率排序
            subset = subset.sort_values(by='false_percentage')

            # 绘制曲线
            ax.plot(subset['false_percentage'], subset['final_e2e_max'],
                    marker='o', label=f'jitter={per_jitter}', color=colors[j])

        ax.set_title(f'num_tasks = {num_tasks}')
        ax.set_xlabel('Failure Rate (false_percentage)')
        ax.set_ylabel('E2E Max (final_e2e_max)')
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig(e2e_plot_name)
    # print(f"图像已保存为：{output_image}")

def plot_histogram_adjust(csv_file,adjust_plot_name):
    num_tasks_to_r_values = {}
    r_exceed_count = 0  # Counter for R values exceeding 1.0
    adjust_success_count = 0
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
                    r_exceed_count += 1
            if row.get('adjust') == 'True':  # Assuming there is a column 'adjust' in the CSV
                adjust_success_count += 1
                
    if total_rows == 0:
        print("No data found.")
        return

    R_exceed_percentage = r_exceed_count / total_rows * 100 if total_rows > 0 else 0
    adjust_success_percentage = (adjust_success_count / total_rows) * 100 if total_rows > 0 else 0

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

        ax.set_title(f"num_tasks = {num_tasks} (per_jitter=20%) - Data Count: {len(r_values)}")
        ax.set_xlabel(f"R_exceed_percentage = {percentage_greater_than_1:.2f}%  "
                    f"adjust_success_percentage = {adjust_success_percentage:.2f}%")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True)

    for idx in range(num_num_tasks, num_rows * num_columns):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.suptitle(f"Distribution of R values for different num_tasks (per_jitter=20%) - "
                f"Overall R_exceed_percentage: {R_exceed_percentage:.2f}%", fontsize=16, y=1.05)

    plt.savefig(adjust_plot_name)
    # plt.show()


def compare_plot_histogram(csv_files, compare_plot_histogram_name):
    dfs = [pd.read_csv(file) for file in csv_files]

    dfs = [df[df['per_jitter'] == 0.2] for df in dfs]

    num_tasks_list = sorted(set.union(*[set(df['num_tasks'].unique()) for df in dfs]))

    fig = plt.figure(figsize=(20, 10 * len(num_tasks_list)))
    outer_grid = GridSpec(len(num_tasks_list), len(csv_files), wspace=0.4, hspace=0.4)

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
            ax.bar(bin_centers, counts, width=bin_width, alpha=0.7, align='center', color=colors[idx], label=f'num_tasks={num_tasks} - Data Count: {len(r_values)}')

            label = os.path.dirname(csv_files[file_idx])
            ax.set_title(f"num_tasks = {num_tasks} (per_jitter=20%) - {label}")
            ax.set_xlabel("R values - R_exceed_percentage = {:.2f}%".format(R_exceed_percentage))
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(True)

    plt.savefig(compare_plot_histogram_name)
    # plt.show()

    
def compare_line_chart_from_csv(csv_files, compare_plot_name):

    num_csv_files = len(csv_files)
    num_columns = 2
    num_rows = (num_csv_files + num_columns - 1) // num_columns

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

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

        label = os.path.dirname(csv_file)
        
        grouped_by_num_tasks = df.groupby('num_tasks')

        for num_tasks, group in grouped_by_num_tasks:
            group_sorted = group.sort_values(by='per_jitter')

            per_jitters = group_sorted['per_jitter'] * 100
            false_percentages = group_sorted['false_percentage']

            ax.plot(per_jitters, false_percentages, label=f'num_tasks={num_tasks}', marker='o')

        ax.set_title(f"False Percentage vs. Jitter ({label})")
        ax.set_xlabel("Jitter Percentage (%)")
        ax.set_ylabel("False Percentage (%)")
        ax.legend()
        ax.grid(True)
    
    for idx in range(num_csv_files, num_rows * num_columns):
        axes[idx].axis('off')
    
    # 获取所有子图中最小/最大的 y 值
    y_min = min(ax.get_ylim()[0] for ax in axes if ax.has_data())
    y_max = max(ax.get_ylim()[1] for ax in axes if ax.has_data())
    for ax in axes:
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(compare_plot_name)
    # plt.show()


def ratio_histogram_from_csv(csv_file, ratio_R_plot_name):
    num_tasks_to_r_values = {}
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            per_jitter = float(row['per_jitter'])
            r_value = float(row['R']) if row['R'] else None
            ratio = float(row['ratios'])
            num_tasks = int(row['num_tasks'])
            
            if per_jitter == 0.2 and ratio == 2 and r_value is not None:
                if num_tasks not in num_tasks_to_r_values:
                    num_tasks_to_r_values[num_tasks] = []
                num_tasks_to_r_values[num_tasks].append(r_value)

    num_num_tasks = len(num_tasks_to_r_values)
    num_columns = 2  
    num_rows = (num_num_tasks + num_columns - 1) // num_columns

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, num_num_tasks)) 
    TOLERANCE = 1e-9
    for idx, (num_tasks, r_values) in enumerate(num_tasks_to_r_values.items()):
        ax = axes[idx]
        num_bins = 50
        bin_range = (0, 1.05)
        bin_width = (bin_range[1] - bin_range[0]) / num_bins

        counts, bin_edges = np.histogram(r_values, bins=num_bins, range=bin_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, counts, width=bin_width, alpha=0.7, align='center', color=colors[idx], label=f'num_tasks={num_tasks}')

        data_count = len(r_values)
        r_exceed_count = len([r for r in r_values if r > (1 + TOLERANCE)])
        R_exceed_percentage = (r_exceed_count / data_count) * 100

        ax.set_title(f"num_tasks = {num_tasks} (ratio=2, per_jitter=0.2) - Data Count: {data_count}")
        ax.set_xlabel("R_exceed_percentage = {:.2f}%".format(R_exceed_percentage))
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True)

    for idx in range(num_num_tasks, num_rows * num_columns):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(ratio_R_plot_name)
    # plt.show()

def ratio_line_chart_from_csv(csv_file, ratio_percent_plot_name):
    ratio_to_false_percentage = {}
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            per_jitter = float(row['per_jitter'])
            false_percentage = float(row['false_percentage'])
            ratio = float(row['ratios'])
            num_tasks = int(row['num_tasks'])
            
            if ratio not in ratio_to_false_percentage:
                ratio_to_false_percentage[ratio] = {}
            if num_tasks not in ratio_to_false_percentage[ratio]:
                ratio_to_false_percentage[ratio][num_tasks] = {}
            if per_jitter not in ratio_to_false_percentage[ratio][num_tasks]:
                ratio_to_false_percentage[ratio][num_tasks][per_jitter] = []
            ratio_to_false_percentage[ratio][num_tasks][per_jitter].append(false_percentage)

    num_ratios = len(ratio_to_false_percentage)
    num_columns = 2  
    num_rows = (num_ratios + num_columns - 1) // num_columns

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for idx, (ratio, num_tasks_to_per_jitter_to_false_percentage) in enumerate(ratio_to_false_percentage.items()):
        ax = axes[idx]
        for num_tasks, per_jitter_to_false_percentage in num_tasks_to_per_jitter_to_false_percentage.items():
            per_jitters = sorted(per_jitter_to_false_percentage.keys())
            false_percentages = [np.mean(per_jitter_to_false_percentage[per_jitter]) for per_jitter in per_jitters]
            ax.plot(per_jitters, false_percentages, label=f'num_tasks={num_tasks}', marker='o')
        
        ax.set_title(f"False Percentage vs. Jitter for Ratio = {ratio}")
        ax.set_xlabel("Jitter Percentage (%)")
        ax.set_ylabel("False Percentage (%)")
        ax.legend()
        ax.grid(True)

    for idx in range(num_ratios, num_rows * num_columns):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(ratio_percent_plot_name)
    # plt.show()

def ratio_for_num_chains(csv_file, ratio_plot_name):
    df = pd.read_csv(csv_file)

    grouped_by_num_tasks = df.groupby('num_tasks')

    num_tasks = len(grouped_by_num_tasks)
    num_columns = 2  
    num_rows = (num_tasks + num_columns - 1) // num_columns

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for idx, (num_tasks, task_group) in enumerate(grouped_by_num_tasks):
        ax = axes[idx]

        grouped_by_ratios = task_group.groupby('ratios')

        for ratio, ratio_group in grouped_by_ratios:
            sorted_group = ratio_group.sort_values(by='per_jitter')
            per_jitters = sorted_group['per_jitter'] * 100
            false_percentages = sorted_group['false_percentage']

            ax.plot(per_jitters, false_percentages, label=f"Ratio = {ratio:.2f}", marker='o')

        ax.set_title(f"num_tasks = {num_tasks}")
        ax.set_xlabel("Jitter Percentage (%)")
        ax.set_ylabel("False Percentage (%)")
        ax.legend()
        ax.grid(True)

    for idx in range(num_tasks, num_rows * num_columns):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(ratio_plot_name)
    # plt.show()


def plot_percent_order(order_file_name, csv_file):
    # Load data from CSV
    df = pd.read_csv(csv_file)
    
    # Extract unique values
    chain_types = df['chain_type'].unique()
    num_tasks_values = df['num_tasks'].unique()
    per_jitter_values = df['per_jitter'].unique()

    fig, axs = plt.subplots(1, len(chain_types), figsize=(15, 5), sharey=True)

    for i, chain_type in enumerate(chain_types):
        ax = axs[i]
        for num_tasks in num_tasks_values:
            subset = df[(df['chain_type'] == chain_type) & (df['num_tasks'] == num_tasks)]
            subset = subset.sort_values(by='per_jitter')
            ax.plot(subset['per_jitter'], subset['false_percentage'], marker='o', label=f'num_tasks={num_tasks}')
        
        ax.set_title(f'Failure Rates for {chain_type}')
        ax.set_xlabel('Jitter')
        ax.set_ylabel('Failure Rate')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(order_file_name)
    print(f"Plot generated and saved to {order_file_name}")
    # plt.close()

def type_percent_order(type_order_file_name, csv_file):
    # Load data from CSV
    df = pd.read_csv(csv_file)
    
    # Extract unique values
    chain_types = df['chain_type'].unique()
    num_tasks_values = df['num_tasks'].unique()
    per_jitter_values = df['per_jitter'].unique()
    
    fig, axs = plt.subplots(1, len(num_tasks_values), figsize=(15, 5), sharey=True)

    if len(num_tasks_values) == 1:
        axs = [axs]  # 如果只有一个子图，确保 axs 是可迭代的
    
    for i, num_tasks in enumerate(num_tasks_values):
        ax = axs[i]
        for chain_type in chain_types:
            subset = df[(df['num_tasks'] == num_tasks) & (df['chain_type'] == chain_type)]
            subset = subset.sort_values(by='per_jitter')
            ax.plot(subset['per_jitter'], subset['false_percentage'], marker='o', label=f'chain_type={chain_type}')
        
        ax.set_title(f'Failure Rates for num_tasks={num_tasks}')
        ax.set_xlabel('Jitter')
        ax.set_ylabel('Failure Rate')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(type_order_file_name)
    print(f"Plot generated and saved to {type_order_file_name}")
    # plt.close()

def plot_r_histogram_order(order_r_plot_name, csv_file):
    # Load data from CSV
    df = pd.read_csv(csv_file)

    # Convert R column to numeric, forcing errors to NaN
    df['R'] = pd.to_numeric(df['R'], errors='coerce')

    # Extract unique values
    chain_types = df['chain_type'].unique()
    num_tasks_values = df['num_tasks'].unique()
    per_jitter_values = df['per_jitter'].unique()
    TOLERANCE = 1e-9
    # Create subplots
    fig, axs = plt.subplots(len(num_tasks_values), len(chain_types), figsize=(20, 10), sharey=True)
    
    # Define colors for different per_jitter values
    colors = plt.cm.viridis(np.linspace(0, 1, len(per_jitter_values)))
    handles = []
    labels = []
    for i, num_tasks in enumerate(num_tasks_values):
        for j, chain_type in enumerate(chain_types):
            ax = axs[i, j]
            r_values_all_jitter = []
            for k, per_jitter in enumerate(per_jitter_values):
                subset = df[(df['chain_type'] == chain_type) & (df['num_tasks'] == num_tasks) & (df['per_jitter'] == per_jitter)]
                r_values = subset['R'].dropna()
                # r_values = subset['R'].dropna().to_numpy(dtype=float)
                # r_values_all_jitter = np.array(r_values_all_jitter, dtype=float)
                r_values_all_jitter.extend(r_values)
                # Plot histogram for each per_jitter with different color
                hist = ax.hist(r_values, bins=20, alpha=0.5, color=colors[k])
                
                # Only add the first handle for each per_jitter to the legend
                if i == 0 and j == 0:
                    handles.append(hist[-1][0])
                    labels.append(f'per_jitter={per_jitter}')

                # # Plot histogram for each per_jitter with different color
                # ax.hist(r_values, bins=20, alpha=0.5, color=colors[k], label=f'per_jitter={per_jitter}')
            
            # Calculate the total number of samples
            total_samples = len(r_values_all_jitter)
            
            # Calculate the number of samples where R > 1
            r_exceed_count = 0
            for r in r_values_all_jitter:
                if r > 1 + TOLERANCE:
                    r_exceed_count += 1
                    print(f"Warning: R value {r} exceeds 1.0 for num_tasks={num_tasks}, chain_type={chain_type}, per_jitter={per_jitter}. This may indicate an error in the data.")
            # Calculate the percentage of R values greater than 1
            if total_samples > 0:
                r_exceed_percentage = (r_exceed_count / total_samples) * 100
            else:
                r_exceed_percentage = 0.0

            # Set title with the percentage of R values greater than 1 and total sample count
            ax.set_title(f'num_tasks={num_tasks}, {chain_type}, - Data Count: {total_samples}')
            ax.set_xlabel(f'R Value, R_exceed_percentage = {r_exceed_percentage:.2f}%')
            ax.set_ylabel('Frequency')
            # ax.legend()
            ax.grid(True)

    plt.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 1.05), title="Legend", fontsize=8)

    plt.tight_layout()
    plt.savefig(order_r_plot_name)
    print(f"Plot generated and saved to {order_r_plot_name}")
    
def type_percent_order_boxplot(box_order_file_name, csv_file):
    df = pd.read_csv(csv_file)
    
    # Check if necessary columns exist
    required_columns = ['chain_type', 'num_tasks', 'false_percentage']
    if not all(column in df.columns for column in required_columns):
        raise ValueError("CSV file is missing required columns.")
    
    # Extract unique values
    chain_types = df['chain_type'].unique()
    num_tasks_values = df['num_tasks'].unique()
    
    # Create subplots
    fig, axs = plt.subplots(1, len(num_tasks_values), figsize=(15, 5), sharey=True)
    
    # Ensure axs is iterable
    if len(num_tasks_values) == 1:
        axs = [axs]
    
    # Define colors for different chain_types
    colors = plt.cm.rainbow(np.linspace(0, 1, len(chain_types)))
    
    # Prepare data for boxplot
    for i, num_tasks in enumerate(num_tasks_values):
        ax = axs[i]
        
        # Extract data for each chain_type and num_tasks
        boxplot_data = []
        labels = []
        for chain_type in chain_types:
            subset = df[(df['num_tasks'] == num_tasks) & (df['chain_type'] == chain_type)]
            boxplot_data.append(subset['false_percentage'].dropna().values)
            labels.append(chain_type)
        
        # Plot boxplot with different colors
        bplot = ax.boxplot(boxplot_data, labels=labels, patch_artist=True)  # patch_artist=True to fill with color
        
        # Set colors for each box
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        
        # Set titles and labels
        ax.set_title(f'Failure Rates for num_tasks={num_tasks}')
        ax.set_xlabel('Chain Type')
        ax.set_ylabel('Failure Rate')
        ax.grid(True)
    
    # Create a legend with colors
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(chain_types))]
    fig.legend(handles, chain_types, title="Chain Types", loc='upper right', bbox_to_anchor=(1.1, 1.05))
    
    
    plt.tight_layout()
    plt.savefig(box_order_file_name)
    print(f"Plot generated and saved to {box_order_file_name}")


def plot_false_percentage_rw(csv_file, RW_plot_false_name):
    # 结构: {num_tasks: {per_jitter: {'read':[...], 'write':[...]}}}
    data = {}

    # 1. 读 CSV
    with open(csv_file, newline='') as f:
        for row in csv.DictReader(f):
            nt   = int(row['num_tasks'])
            jit  = float(row['per_jitter'])
            fr   = float(row['false_percentage_read'])
            fw   = float(row['false_percentage_write'])

            data.setdefault(nt, {}).setdefault(jit, {'read': [], 'write': []})
            data[nt][jit]['read'].append(fr)
            data[nt][jit]['write'].append(fw)

    # 2. 准备画布
    plt.figure(figsize=(10, 6))

    # 3. 画线：先写后读，避免图例顺序混乱
    for nt in sorted(data):
        j_sorted = sorted(data[nt])
        x = [j*100 for j in j_sorted]

        y_write = [np.mean(data[nt][j]['write'])*100 for j in j_sorted]
        y_read  = [np.mean(data[nt][j]['read'])*100  for j in j_sorted]

        # 写：实线
        plt.plot(x, y_write,
                 marker='o', linestyle='-', linewidth=2,
                 label=f'write (num_tasks={nt})')

        # 读：虚线
        plt.plot(x, y_read,
                 marker='s', linestyle='--', linewidth=2,
                 label=f'read  (num_tasks={nt})')

    # 4. 细节
    plt.title('Read/Write False Percentage vs. Jitter')
    plt.xlabel('Jitter Percentage (%)')
    plt.ylabel('False Percentage (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 5. 保存
    plt.savefig(RW_plot_false_name, dpi=300)
    print(f'Saved to {RW_plot_false_name}')
    # plt.show()



def plot_histogram_rw_from_csv(csv_file, R_plot_name_RW):
    JITTER_FILTER = 0.2          # 只画 20% 抖动
    TOLERANCE = 1e-9

    # 1. 收集数据：{num_tasks: {'write': [...], 'read': [...]}}
    data = {}
    r_exceed_write = 0
    r_exceed_read  = 0
    total_rows = 0

    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            if float(row['per_jitter']) != JITTER_FILTER:
                continue

            nt = int(row['num_tasks'])
            rw = float(row['R_write']) if row['R_write'].strip() != '' else None
            rr = float(row['R_read'])  if row['R_read'].strip()  != '' else None

            data.setdefault(nt, {'write': [], 'read': []})
            if rw is not None:
                data[nt]['write'].append(rw)
                if rw > 1 + TOLERANCE:
                    r_exceed_write += 1
            if rr is not None:
                data[nt]['read'].append(rr)
                if rr > 1 + TOLERANCE:
                    r_exceed_read += 1

    if not data:
        print("No data for per_jitter = 0.2")
        return

    # 2. 得到实际任务数
    sorted_tasks = sorted(data.keys())
    n_tasks = len(sorted_tasks)

    # 3. 动态画布：2 行 × n_tasks 列
    fig, axes = plt.subplots(2, n_tasks,
                             figsize=(4 * n_tasks, 8),
                             sharex=True, sharey=False)
    if n_tasks == 1:
        axes = axes.reshape(2, 1)   # 兼容 1 列时的退化情况
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, n_tasks))

    # 4. 画写（上半行）
    for i, nt in enumerate(sorted_tasks):
        ax = axes[i]
        vals = [v for v in data[nt]['write'] if v is not None]
        n_vals = len(vals) 
        counts, edges = np.histogram(vals, bins=50, range=(0, 1.05))
        centers = (edges[:-1] + edges[1:]) / 2
        ax.bar(centers, counts,
               width=edges[1]-edges[0],
               color=colors[i], alpha=0.7)
        pct = (np.array(vals) > 1).mean() * 100
        ax.set_title(f'Write,num_tasks={nt}\ncount={n_vals},R>{1:.0f}: {pct:.1f}%')
        ax.set_ylabel('Frequency')
        ax.grid(True)

    # 5. 画读（下半行）
    for i, nt in enumerate(sorted_tasks):
        ax = axes[n_tasks + i]
        vals = [v for v in data[nt]['read'] if v is not None]
        n_vals = len(vals)                      # 样本数量
        counts, edges = np.histogram(vals, bins=50, range=(0, 1.05))
        centers = (edges[:-1] + edges[1:]) / 2
        ax.bar(centers, counts,
               width=edges[1]-edges[0],
               color=colors[i], alpha=0.7)
        pct = (np.array(vals) > 1).mean() * 100
        ax.set_title(f'Read, num_tasks={nt}\ncount={n_vals},R>{1:.0f}: {pct:.1f}%')
        ax.set_xlabel('R value')
        ax.grid(True)

    # # 隐藏多余子图
    # for j in range(len(sorted_tasks) + 4, max_subplots):
    #     axes[j].axis('off')

    plt.suptitle(
        f'Distribution of R_write (left) and R_read (right) at per_jitter={int(JITTER_FILTER*100)}%\n'
        f'Global exceed 1.0: write={r_exceed_write}, read={r_exceed_read} (total rows {total_rows})',
        fontsize=16, y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(R_plot_name_RW, dpi=300)
    print(f'Saved to {R_plot_name_RW}')
    # plt.show()



def plot_baseline_from_csv(csv_file, baseline_plot_name):
    df = pd.read_csv(csv_file)

    # 将 per_jitter 转成百分比（可选，仅用于展示）
    df['per_jitter_pct'] = df['per_jitter'] * 100

    # 计算基准：jitter=0 时的 max_reaction_time
    baseline = (
        df[df['per_jitter'] == 0.0]
        .groupby('num_tasks')['max_reaction_time']
        .mean()
        .rename('baseline')
        .reset_index()
    )

    # 合并基准，计算 ratio
    df = df.merge(baseline, on='num_tasks')
    df['ratio'] = df['max_reaction_time'] / df['baseline']

    # 绘图
    plt.figure(figsize=(6, 4))
    sns.lineplot(
        data=df,
        x='per_jitter_pct',
        y='ratio',
        hue='num_tasks',
        marker='o',
        palette='tab10'
    )

    # 画 y=1 参考线
    plt.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='baseline (jitter=0)')

    plt.xlabel("Jitter (%)")
    plt.ylabel("Ratio  e2e(jitter) / e2e(0)")
    plt.title("End-to-End Reaction Time Ratio vs Jitter")
    plt.legend(title='num_tasks', ncol=2, fontsize=8, title_fontsize=8, markerscale=0.7, labelspacing=0.3,  handletextpad=0.3 )
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"{baseline_plot_name}", dpi=300)
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot histograms from a CSV file.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing the data.")
    # parser.add_argument("R_plot_name", type=str, help="Name of the output plot file for R values.")
    # parser.add_argument("percent_plot_name", type=str, help="Name of the output plot file for false percentages.")
    # parser.add_argument("e2e_plot_name", type=str, help="Name of the output plot file for final e2e max values.")

    # parser.add_argument("adjust_plot_name", type=str, help="Name of the output plot file for adjusted R values.")
    
    # parser.add_argument("csv_files", type=str, nargs='+', help="Paths to the CSV files containing the data.")
    # parser.add_argument("compare_plot_histogram_name", type=str, help="Name of the output plot file for compare plot.")
    # parser.add_argument("compare_plot_name", type=str, help="Name of the output plot file for compare plot.")

    # parser.add_argument("ratio_R_plot_name", type=str)
    # parser.add_argument("ratio_percent_plot_name", type=str)
    # parser.add_argument("ratio_plot_name", type=str)

    # parser.add_argument("order_file_name", type=str, help="Name of the output plot file for percent order. ")
    # parser.add_argument("type_order_file_name", type=str, help="Name of the output plot file for type order. ")
    # parser.add_argument("order_r_plot_name", type=str, help="Name of the output plot file for R histogram order. ")
    # parser.add_argument("box_order_file_name", type=str, help="Name of the output plot file for box order. ")

    # parser.add_argument("plot_histogram_rw_name", type=str, help="Name of the output plot file for histogram RW.")
    # parser.add_argument("plot_false_percentage_rw_name", type=str, help="Name of the output plot file for false percentage RW.")
    parser.add_argument("plot_baseline_name", type=str, help="Name of the output plot file for baseline.")

    args = parser.parse_args()

    # plot_histogram_from_csv(args.csv_file, args.R_plot_name)
    # plot_line_chart_from_csv(args.csv_file, args.percent_plot_name)
    # plot_final_e2e_max(args.e2e_plot_name, args.csv_file)

    # plot_histogram_adjust(args.csv_file, args.adjust_plot_name)

    # compare_plot_histogram(args.csv_files, args.compare_plot_histogram_name)
    # compare_line_chart_from_csv(args.csv_files, args.compare_plot_name)
    
    # ratio_histogram_from_csv(args.csv_file, args.ratio_R_plot_name)
    # ratio_line_chart_from_csv(args.csv_file, args.ratio_percent_plot_name)
    # ratio_for_num_chains(args.csv_file, args.ratio_plot_name)

    # plot_percent_order(args.order_file_name, args.csv_file)
    # type_percent_order(args.type_order_file_name, args.csv_file)
    # plot_r_histogram_order(args.order_r_plot_name, args.csv_file)
    # type_percent_order_boxplot(args.box_order_file_name, args.csv_file)

    # plot_histogram_rw_from_csv(args.csv_file, args.plot_histogram_rw_name)
    # plot_false_percentage_rw(args.csv_file, args.plot_false_percentage_rw_name)

    plot_baseline_from_csv(args.csv_file, args.plot_baseline_name)