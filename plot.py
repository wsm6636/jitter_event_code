import matplotlib.pyplot as plt
import csv
import numpy as np
import argparse 
from scipy.stats import gaussian_kde
import pandas as pd
import ast

def plot_histogram_from_csv(csv_file,R_plot_name):
    jitter_to_r_values = {}
    r_exceed_count = 0  # Counter for R values exceeding 1.0
    adjust_success_count = 0
    total_rows = 0 
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        # r_values_count = 0  
        for row in reader:
            total_rows += 1 
            per_jitter = float(row['per_jitter'])
            r_value = float(row['R']) if row['R'] else None
            if r_value is not None:
                if per_jitter not in jitter_to_r_values:
                    jitter_to_r_values[per_jitter] = []
                jitter_to_r_values[per_jitter].append(r_value)
                if r_value > 1:
                    print(f"Warning: R value {r_value} exceeds 1.0 for per_jitter={per_jitter}. This may indicate an error in the data.")
                    r_exceed_count += 1  


                
    R_exceed_percentage = r_exceed_count / total_rows * 100 if total_rows > 0 else 0

    plt.figure(figsize=(12, 8))
    num_bins = 50
    bin_range = (0, 1.05)  
    bin_width = (bin_range[1] - bin_range[0]) / num_bins

    jitters = sorted(jitter_to_r_values.keys())
    num_groups = len(jitters)
    group_width = bin_width / (num_groups + 1)  
    # print(f"Number of valid R values: {r_values_count}")  
    for idx, per_jitter in enumerate(jitters):
        r_values = jitter_to_r_values[per_jitter]
        
        # Grouped Histogram
        offset = (idx - num_groups / 2) * group_width
        counts, bin_edges = np.histogram(r_values, bins=num_bins, range=bin_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.bar(bin_centers + offset, counts, width=group_width, alpha=0.3, align='center')

        # trend line
        if len(r_values) > 1:  
            kde = gaussian_kde(r_values)
            x = np.linspace(bin_range[0], bin_range[1], 1000)
            y = kde(x)
            plt.plot(x, y * len(r_values) * bin_width, linestyle='-', label=f'per_jitter={per_jitter}',color=plt.cm.tab10(idx))
        else:
            print(f"Skipping KDE for per_jitter={per_jitter} due to insufficient data points.")
    

    plt.title("Distribution of R values for different Jitter Percentages")
    plt.xlabel("R_exceed_percentage = {:.2f}%".format(R_exceed_percentage))
    plt.ylabel("Frequency")
    plt.xlim(bin_range)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{R_plot_name}")
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
    num_columns = 2  # 每行显示 2 个子图
    num_rows = (num_num_tasks + num_columns - 1) // num_columns

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, num_num_tasks))  # 生成不同的颜色

    for idx, (num_tasks, r_values) in enumerate(num_tasks_to_r_values.items()):
        ax = axes[idx]
        num_bins = 50
        bin_range = (0, 1.05)
        bin_width = (bin_range[1] - bin_range[0]) / num_bins

        counts, bin_edges = np.histogram(r_values, bins=num_bins, range=bin_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, counts, width=bin_width, alpha=0.7, align='center', color=colors[idx], label=f'num_tasks={num_tasks}')

        data_count = len(r_values)
        ax.set_title(f"num_tasks = {num_tasks} (ratio=2, per_jitter=0.2) - Data Count: {data_count}")
        ax.set_xlabel("R = max_reaction_time / final_e2e_max")
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
    num_columns = 2  # 每行显示 2 个子图
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

    # 按 num_tasks 分组
    grouped_by_num_tasks = df.groupby('num_tasks')

    # 计算子图的行列数
    num_tasks = len(grouped_by_num_tasks)
    num_columns = 2  # 每行显示 2 个子图
    num_rows = (num_tasks + num_columns - 1) // num_columns

    # 创建子图
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for idx, (num_tasks, task_group) in enumerate(grouped_by_num_tasks):
        ax = axes[idx]

        # 按 ratios 分组
        grouped_by_ratios = task_group.groupby('ratios')

        for ratio, ratio_group in grouped_by_ratios:
            # 按 per_jitter 排序
            sorted_group = ratio_group.sort_values(by='per_jitter')
            per_jitters = sorted_group['per_jitter'] * 100
            false_percentages = sorted_group['false_percentage']

            # 绘制折线图
            ax.plot(per_jitters, false_percentages, label=f"Ratio = {ratio:.2f}", marker='o')

        # 设置子图标题和标签
        ax.set_title(f"num_tasks = {num_tasks}")
        ax.set_xlabel("Jitter Percentage (%)")
        ax.set_ylabel("False Percentage (%)")
        ax.legend()
        ax.grid(True)

    # 隐藏多余的子图
    for idx in range(num_tasks, num_rows * num_columns):
        axes[idx].axis('off')

    # 调整子图间距
    plt.tight_layout()
    plt.savefig(ratio_plot_name)
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot histograms from a CSV file.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing the data.")
    parser.add_argument("R_plot_name", type=str, help="Name of the output plot file for R values.")
    parser.add_argument("percent_plot_name", type=str, help="Name of the output plot file for false percentages.")
    # parser.add_argument("ratio_R_plot_name", type=str)
    # parser.add_argument("ratio_percent_plot_name", type=str)
    # parser.add_argument("ratio_plot_name", type=str)


    args = parser.parse_args()

    plot_histogram_from_csv(args.csv_file, args.R_plot_name)
    plot_line_chart_from_csv(args.csv_file, args.percent_plot_name)
    # ratio_histogram_from_csv(args.csv_file, args.ratio_R_plot_name)
    # ratio_line_chart_from_csv(args.csv_file, args.ratio_percent_plot_name)
    # ratio_for_num_chains(args.csv_file, args.ratio_plot_name)
    # print(f"Plots generated and saved to {args.ratio_R_plot_name}, {args.ratio_percent_plot_name}, {args.ratio_plot_name}")