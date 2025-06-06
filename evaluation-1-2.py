# 文件名：main.py
import csv
import datetime
from analysis import run_analysis
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from plot import ratio_histogram_from_csv
from plot import ratio_line_chart_from_csv
from plot import ratio_for_num_chains
from analysis import RandomEvent
import random
import time


def generate_periods_and_offsets(selected_periods):

    selected_read_offsets = [random.uniform(0, (period - 1)) for period in selected_periods]  
    selected_write_offsets = [read_offset + period for read_offset, period in zip(selected_read_offsets, selected_periods)]

    return selected_read_offsets, selected_write_offsets


def generate_periods(ratio, num_tasks, min_period, max_period):

    periods = [random.uniform(min_period, max_period)]  

    # 生成剩余的周期
    for _ in range(1, num_tasks):
        # 确保新周期与前一个周期的比值不小于 ratio
        last_period = periods[-1]
        count = 0
        while True:
            new_period = random.uniform(min_period, max_period)
            r = new_period / last_period
            count += 1
            if r >= ratio:
                break
            if count > 1000:  # 防止无限循环
                if new_period * ratio >= max_period:  # 防止无限循环
                    new_period = last_period * ratio
                    break
            
                
        # 添加新周期到列表  
        periods.append(new_period)
        
    return periods



def main():
    # INCREASE here to have more experiments per same settings
    num_repeats =20  # number of repetitions: if 10 takes about 20 minutes on Shumo's laptop
    # Enrico's laptop: num_repeats=10 ==> 32 seconds

    
    # jitters = [0,0.01,0.02,0.05,0.1,0.2,0.5,1]  # maxjitter = percent jitter * period
    jitters = [0,0.02,0.05,0.1,0.2,0.3,0.4,0.5]  # maxjitter = percent jitter * period

    num_chains = [3,5,8,10] 
    # num_chains  = [3,5]  # for test

    # below we are setting the random seed. Depending on the need, it may be set to a fixed value or a time-dependent value
    # RANDOM SEED: set it to time to avoid repetition. Or to a given value for reproducibility
    random_seed = int(time.time())
    # random_seed = 100  # fixed seed

    min_period = 1  # minimum period
    max_period = 1000  # maximum period

    ratios = np.arange(1.0, 3.0, 0.5)
    print(f"Ratios: {ratios}")
    # name for log file
    timestamp = datetime.datetime.fromtimestamp(random_seed).strftime("%Y%m%d_%H%M%S")
    percent_plot_name = f"percent_{num_repeats}_{timestamp}.png"
    R_plot_name = f"R_{num_repeats}_{timestamp}.png"
    results_csv = f"data_{num_repeats}_{timestamp}.csv" 
    ratio_plot_name = f"ratios_{num_repeats}_{timestamp}.png"

    # preparing list for storing result
    results = {num_tasks: {per_jitter: {ratio: [] for ratio in ratios} for per_jitter in jitters} for num_tasks in num_chains}
    final = {num_tasks: {per_jitter: {ratio: [] for ratio in ratios} for per_jitter in jitters} for num_tasks in num_chains}
    false_results = {num_tasks: {per_jitter: {ratio: 0 for ratio in ratios} for per_jitter in jitters} for num_tasks in num_chains}

    # TODO: add random_seed to the filename
    # run analysis
    for ratio in ratios:
        for i in range(num_repeats):            # loop on number of repetitions
            random.seed(random_seed)        
            for num_tasks in num_chains:        # on number of tasks in a chain
                selected_periods = generate_periods(ratio, num_tasks, min_period, max_period)
                selected_read_offsets, selected_write_offsets = generate_periods_and_offsets(selected_periods)

                print(f"================== selected_period {selected_periods}, ratio {ratio} ==================")
                for per_jitter in jitters:      # on relative (to period) magnitude of jitter
                    # generate the jitter
                    # only generate the jitter
                    print(f"================== num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} random_seed {random_seed} ==================")
                    final_e2e_max, max_reaction_time,  final_r, final_w, tasks = run_analysis(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter)
                    # value of rate "= max_reaction_time / final_e2e_max"
                    if final_e2e_max != 0:
                        r = max_reaction_time / final_e2e_max
                    else:
                        r = None
                        false_results[num_tasks][per_jitter][ratio] += 1  # algorithm failed

                    results[num_tasks][per_jitter][ratio].append((final_e2e_max, max_reaction_time,r,tasks))
                    final[num_tasks][per_jitter][ratio].append((final_r, final_w))
                    
            random_seed = random_seed+1

    # algorithm2 failed percentage 
    for ratio in ratios:
        for num_tasks in num_chains:
            for per_jitter in jitters:
                false_percentage = (false_results[num_tasks][per_jitter][ratio] / num_repeats)
                false_results[num_tasks][per_jitter][ratio] = false_percentage

    # save results to csv
    with open(results_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["num_tasks", "per_jitter", "final_e2e_max", "max_reaction_time", "R", "false_percentage","ratios"])
        for ratio in ratios:
            for num_tasks in num_chains:
                for per_jitter in jitters:
                    false_percentage = false_results[num_tasks][per_jitter][ratio]
                    for (final_e2e_max, max_reaction_time, r, _) in results[num_tasks][per_jitter][ratio]:
                        writer.writerow([num_tasks, per_jitter, final_e2e_max, max_reaction_time, r, false_percentage, ratio])

    print(f"All results saved to {results_csv}")

    # plotting: uncomment to have plots made automatically
    ratio_histogram_from_csv(results_csv, R_plot_name)
    print(f"Plots generated and saved to {R_plot_name}")
    ratio_line_chart_from_csv(results_csv, percent_plot_name)
    print(f"Plots generated and saved to {percent_plot_name}")
    ratio_for_num_chains(results_csv, ratio_plot_name)
    print(f"Plots generated and saved to {ratio_plot_name}")

if __name__ == "__main__":
    main()
    