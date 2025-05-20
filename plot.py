import matplotlib.pyplot as plt
import csv
import numpy as np
import argparse  # 导入 argparse 模块用于处理命令行参数
from scipy.stats import gaussian_kde

def plot_histogram_from_csv(csv_file):
    jitter_to_r_values = {}
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            per_jitter = float(row['per_jitter'])
            r_value = float(row['R']) if row['R'] else None
            if r_value is not None:
                if per_jitter not in jitter_to_r_values:
                    jitter_to_r_values[per_jitter] = []
                jitter_to_r_values[per_jitter].append(r_value)
                if r_value > 1:
                    print(f"Warning: R value {r_value} exceeds 1.0 for per_jitter={per_jitter}. This may indicate an error in the data.")

    # 绘制分组直方图和趋势曲线
    plt.figure(figsize=(12, 8))
    num_bins = 15
    bin_range = (0, 1)  # 设置直方图的范围
    bin_width = (bin_range[1] - bin_range[0]) / num_bins

    # 获取所有 per_jitter 的值
    jitters = sorted(jitter_to_r_values.keys())
    num_groups = len(jitters)
    group_width = bin_width / (num_groups + 1)  # 每个组的宽度

    # 绘制每个组的直方图和趋势曲线
    for idx, per_jitter in enumerate(jitters):
        r_values = jitter_to_r_values[per_jitter]
        #分组
        offset = (idx - num_groups / 2) * group_width
        counts, bin_edges = np.histogram(r_values, bins=num_bins, range=bin_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.bar(bin_centers + offset, counts, width=group_width, alpha=0.5, label=f'per_jitter={per_jitter}', align='center')

        #重叠
        # counts, bin_edges = np.histogram(r_values, bins=num_bins, range=bin_range)
        # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # plt.bar(bin_centers, counts, width=bin_width, alpha=0.3, label=f'per_jitter={per_jitter}',align='center', color=plt.cm.tab10(idx))

        # 绘制阶梯直方图
        # counts, bin_edges = np.histogram(r_values, bins=num_bins, range=bin_range)
        # plt.stairs(counts, bin_edges, alpha=1, label=f'per_jitter={per_jitter}', color=plt.cm.tab10(idx))

        # #趋势
        # if len(r_values) > 1:  # 确保数据集中有足够的数据点
        #     kde = gaussian_kde(r_values)
        #     x = np.linspace(bin_range[0], bin_range[1], 1000)
        #     y = kde(x)
        #     # # 放大趋势线的 y 轴值，使其比直方图更高
        #     # max_count = max(counts) if counts.size > 0 else 1
        #     # y = y * max_count
        #     plt.plot(x, y * len(r_values) * bin_width, linestyle='-', label=f'per_jitter={per_jitter}',color=plt.cm.tab10(idx))
        # else:
        #     print(f"Skipping KDE for per_jitter={per_jitter} due to insufficient data points.")

    plt.title("Distribution of R Values for Different Jitter Percentages")
    plt.xlabel("R = max_reaction_time / final_e2e_max")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{R_plot_name}")
    plt.show()

    


def plot_line_chart_from_csv(csv_file):
    # 读取 CSV 文件中的数据
    jitter_to_false_percentage = {}
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            per_jitter = float(row['per_jitter'])
            false_percentage = float(row['false_percentage'])  # 使用 false_percentage 字段
            num_tasks = int(row['num_tasks'])  # 使用 num_tasks 字段

            if num_tasks not in jitter_to_false_percentage:
                jitter_to_false_percentage[num_tasks] = {}

            if per_jitter not in jitter_to_false_percentage[num_tasks]:
                jitter_to_false_percentage[num_tasks][per_jitter] = []

            jitter_to_false_percentage[num_tasks][per_jitter].append(false_percentage)

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    for num_tasks, jitter_data in jitter_to_false_percentage.items():
        jitter_percent = [jitter * 100 for jitter in sorted(jitter_data.keys())]
        false_percentages = [np.mean(jitter_data[jitter]) for jitter in sorted(jitter_data.keys())]
        plt.plot(jitter_percent, false_percentages, label=f"num_tasks={num_tasks}", marker='o')

    plt.title("False Percentage vs. Jitter for Different Number of Tasks")
    plt.xlabel("Jitter Percentage (%)")
    plt.ylabel("False Percentage (%)")
    plt.legend()
    plt.grid(True)
    plt.xticks(jitter_percent)  # 设置 x 轴刻度为完整的 jitter_percent
    plt.savefig(f"{percent_plot_name}")
    plt.show()


if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Plot histograms and line charts from a CSV file.")
    parser.add_argument("name", type=str, help="Path to the CSV file containing the data.")
    args = parser.parse_args()

    csv_file = f"{args.name}.csv" 
    percent_plot_name = f"{args.name}_percent.png"
    R_plot_name = f"{args.name}_R.png"

    plot_histogram_from_csv(csv_file)
    plot_line_chart_from_csv(csv_file)