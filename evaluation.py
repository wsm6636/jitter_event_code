# 文件名：main.py
import csv
import datetime
from analysis import run_analysis
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def main():
    num_repeats = 10  # number of repeats
    niter = 1  # Iterations
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000]  # periods
    # jitters = [0,0.01,0.02,0.05,0.1,0.2,0.5,1]  # percent jitter : maxjitter = percent jitter * period
    # num_chains = [3,5,8,10] 

    jitters = [0,0.01,0.02,0.03]  # percent jitter : maxjitter = percent jitter * period
    num_chains = [3,5] 

    # log 
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    percent_plot_name = f"result/percent_{num_repeats}_{niter}_{timestamp}.png"
    R_plot_name = f"result/R_{num_repeats}_{niter}_{timestamp}.png"
    results_csv = f"result/data_{num_repeats}_{niter}_{timestamp}.csv" 

    results = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    final = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    false_results = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}

    # run analysis
    for num_tasks in num_chains:
        for per_jitter in jitters:
            false_counts = 0
            for i in range(num_repeats):
                print(f"================== num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} ==================")
                final_e2e_max, max_reaction_time,  final_r, final_w, tasks = run_analysis(num_tasks, periods, per_jitter, niter)
                # R = max_reaction_time / final_e2e_max
                if final_e2e_max != 0:
                    r = max_reaction_time / final_e2e_max
                else:
                    r = None
                    false_counts += 1  # algorithm2 failed

                results[num_tasks][per_jitter].append((final_e2e_max, max_reaction_time,r,tasks))
                final[num_tasks][per_jitter].append((final_r, final_w))

            # failed percentage
            false_percentage = (false_counts / num_repeats) * 100
            false_results[num_tasks][per_jitter].append(false_percentage)


    # save results to csv
    with open(results_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["num_tasks", "per_jitter", "final_e2e_max", "max_reaction_time", "R", "false_percentage"])
        for num_tasks in num_chains:
            for per_jitter in jitters:
                false_percentage = false_results[num_tasks][per_jitter][0]
                for (final_e2e_max, max_reaction_time, r, _) in results[num_tasks][per_jitter]:
                    writer.writerow([num_tasks, per_jitter, final_e2e_max, max_reaction_time, r, false_percentage])

    print(f"All results saved to {results_csv}")
    
    
    # plot percentage
    plt.figure(figsize=(10, 6))
    for num_tasks in num_chains:
        jitter_percent = [jitter * 100 for jitter in jitters]
        percentages = [false_results[num_tasks][per_jitter][0] for per_jitter in jitters]  
        plt.plot(jitter_percent, percentages, label=f"num_tasks={num_tasks}", marker='o')

    plt.title("False Percentage vs. Jitter for Different Number of Tasks")
    plt.xlabel("Jitter Percentage (%)")
    plt.ylabel("False Percentage (%)")
    plt.legend()
    plt.grid(True)
    plt.xticks(jitter_percent)  
    plt.savefig(f"{percent_plot_name}") 
    plt.show()
    print(f"Percentage plot saved to {percent_plot_name}")


    # Plot the distribution of R values
    plt.figure(figsize=(12, 8))
    num_bins = 20
    bin_range = (0, 1.1) 
    bin_width = (bin_range[1] - bin_range[0]) / num_bins

    jitters = sorted(jitters)
    num_groups = len(jitters)
    group_width = bin_width / (num_groups + 1)  

    for idx, per_jitter in enumerate(jitters):
        all_r_values = []
        for num_tasks in num_chains:
            r_values = [r for final_e2e_max, max_reaction_time, r, _ in results[num_tasks][per_jitter] if r is not None]
            all_r_values.extend(r_values)

        #分组
        offset = (idx - num_groups / 2) * group_width
        counts, bin_edges = np.histogram(r_values, bins=num_bins, range=bin_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.bar(bin_centers + offset, counts, width=group_width, alpha=0.7, label=f'per_jitter={per_jitter}', align='center')

        #重叠
        # counts, bin_edges = np.histogram(r_values, bins=num_bins, range=bin_range)
        # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # plt.bar(bin_centers, counts, width=bin_width, alpha=0.1, align='center', color=plt.cm.tab10(idx))

        # 绘制阶梯直方图
        # counts, bin_edges = np.histogram(r_values, bins=num_bins, range=bin_range)
        # plt.stairs(counts, bin_edges, alpha=1, label=f'per_jitter={per_jitter}', color=plt.cm.tab10(idx))

        #趋势
        # if len(r_values) > 1:  # 确保数据集中有足够的数据点
        #     kde = gaussian_kde(r_values)
        #     x = np.linspace(bin_range[0], bin_range[1], 1000)
        #     y = kde(x)
        #     # 放大趋势线的 y 轴值，使其比直方图更高
        #     max_count = max(counts) if counts.size > 0 else 1
        #     y = y * max_count
        #     plt.plot(x, y * len(r_values) * bin_width, linestyle='-', label=f'per_jitter={per_jitter}',color=plt.cm.tab10(idx))
        # else:
        #     print(f"Skipping KDE for per_jitter={per_jitter} due to insufficient data points.")

    plt.title("Distribution of R Values for Different Jitter Percentages")
    plt.xlabel("R = max_reaction_time / final_e2e_max")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig(R_plot_name)
    plt.show()
    print(f"R distribution plot saved to {R_plot_name}")


if __name__ == "__main__":
    main()