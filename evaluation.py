# 文件名：main.py
import csv
import datetime
from analysis import run_analysis
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from plot import plot_histogram_from_csv 
from plot import plot_line_chart_from_csv

def main():
    num_repeats = 1000  # number of repeats
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000]  # periods
    jitters = [0,0.01,0.02,0.05,0.1,0.2,0.5,1]  # maxjitter = percent jitter * period
    num_chains = [3,5,8,10] 

    # log 
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    percent_plot_name = f"percent_{num_repeats}_{timestamp}.png"
    R_plot_name = f"R_{num_repeats}_{timestamp}.png"
    results_csv = f"data_{num_repeats}_{timestamp}.csv" 

    results = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    final = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    false_results = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    # r_values_count = 0
    # run analysis
    for num_tasks in num_chains:
        for per_jitter in jitters:
            false_counts = 0
            for i in range(num_repeats):
                print(f"================== num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} ==================")
                final_e2e_max, max_reaction_time,  final_r, final_w, tasks = run_analysis(num_tasks, periods, per_jitter)
                # R = max_reaction_time / final_e2e_max
                if final_e2e_max != 0:
                    r = max_reaction_time / final_e2e_max
                    # r_values_count += 1 
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
    
    # print(r_values_count)

    plot_histogram_from_csv(results_csv, R_plot_name)
    print(f"Plots generated and saved to {R_plot_name}")
    plot_line_chart_from_csv(results_csv, percent_plot_name)
    print(f"Plots generated and saved to {percent_plot_name}")



if __name__ == "__main__":
    main()
    