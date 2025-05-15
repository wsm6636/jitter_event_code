# 文件名：main.py
from WATER_period import run_analysis
import numpy as np

def main():
    num_repeats = 10  # 重复次数
    num_tasks = 5  # 任务数量
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000]  # 周期列表
    niter = 10  # 迭代次数
    per_jitter = 0.05
    log = False  # 是否记录日志
    draw_plot = False  # 是否绘制图形

    results = []

    for i in range(num_repeats):
        print(f"================== Repeat {i} ==================")
        final_e2e_max, max_reaction_time = run_analysis(num_tasks, periods, per_jitter, niter, log=False, draw_plot=False)
        results.append((final_e2e_max, max_reaction_time))

    # 打印结果
    print("\n================== Results ==================")
    for i, (final_e2e_max, max_reaction_time) in enumerate(results):
        print(f"Repeat {i}: Final E2E Max: {final_e2e_max:.2f}, Maximized Reaction Time: {max_reaction_time:.2f}")


if __name__ == "__main__":
    main()