# 文件名：main.py
import datetime
from WATER_period import run_analysis
import numpy as np
import matplotlib.pyplot as plt


def main():
    num_repeats = 100  # 重复次数
    niter = 1  # 迭代次数
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000]  # 周期列表
    jitters = [0,0.01,0.02,0.05,0.1,0.2,0.5,1]
    num_chains = [3,5,8,10]

    printlog = True  # 是否打印日志


    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    percent_plot_name = f"result/percent_{num_repeats}_{niter}_{timestamp}.png"
    R_plot_name = f"result/R_{num_repeats}_{niter}_{timestamp}.png"

# 用于存储结果
    results = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    final = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}
    false_results = {num_tasks: {per_jitter: [] for per_jitter in jitters} for num_tasks in num_chains}

# 迭代 num_tasks 和 jitter
    for num_tasks in num_chains:
        for per_jitter in jitters:
            zero_counts = 0
            for i in range(num_repeats):
                print(f"================== num_tasks {num_tasks} per_jitter {per_jitter} Repeat {i} ==================")
                # 运行分析
                final_e2e_max, max_reaction_time,  final_r, final_w, tasks = run_analysis(num_tasks, periods, per_jitter, niter)
                if final_e2e_max != 0:
                    r = max_reaction_time / final_e2e_max
                else:
                    r = None
                    zero_counts += 1
                results[num_tasks][per_jitter].append((final_e2e_max, max_reaction_time,r,tasks))
                final[num_tasks][per_jitter].append((final_r, final_w))

            # 计算 final_e2e_max = 0 的百分比
            zero_percentage = (zero_counts / num_repeats) * 100
            false_results[num_tasks][per_jitter].append(zero_percentage)
            # print(f"Percentage of False: {zero_percentage:.2f}%")


    # 打印结果
    print("\n================== Results ==================")
    for num_tasks in num_chains:
        print(f"==================Number of Tasks: {num_tasks}==================")
        for per_jitter in jitters:
            zero_percentage = false_results[num_tasks][per_jitter][0]  # 取第一个元素
            print(f"per_jitter {per_jitter}: Percentage of False: {zero_percentage:.2f}%")
            for (final_e2e_max, max_reaction_time, r,_) in results[num_tasks][per_jitter]:
                if r is not None:
                    print(f"    Final E2E Max: {final_e2e_max:.2f}, Maximized Reaction Time: {max_reaction_time:.2f}, R = {r:.2f}")
                else:
                    print(f"    Final E2E Max: {final_e2e_max:.2f}, Maximized Reaction Time: {max_reaction_time:.2f}, R = None")

    

    
    # 绘制折线图
    plt.figure(figsize=(10, 6))
    for num_tasks in num_chains:
        # 将 jitters 转换为百分比
        jitter_percent = [jitter * 100 for jitter in jitters]
        percentages = [false_results[num_tasks][per_jitter][0] for per_jitter in jitters]  # 提取每个 per_jitter 对应的 zero_percentage
        plt.plot(jitter_percent, percentages, label=f"num_tasks={num_tasks}", marker='o')


    plt.title("Zero Percentage vs. Jitter for Different Number of Tasks")
    plt.xlabel("Jitter Percentage (%)")
    plt.ylabel("Zero Percentage (%)")
    plt.legend()
    plt.grid(True)
    plt.xticks(jitter_percent)  # 设置 x 轴刻度为完整的 jitter_percent
    # plt.show()
    plt.savefig(f"{percent_plot_name}")  # 保存图像
    print(f"Percentage plot saved to {percent_plot_name}")

    # 绘制所有 r 值的散点图
    plt.figure(figsize=(10, 6))
    index = 0
    for num_tasks in num_chains:
        for per_jitter in jitters:
            r_values = [r for final_e2e_max, max_reaction_time, r, _ in results[num_tasks][per_jitter] if r is not None]
            indices = list(range(index, index + len(r_values)))  # 为每个 r 值分配一个唯一的索引
            index += len(r_values)  # 更新索引
            plt.scatter(indices, r_values, color='blue')
    # 在 y 轴的 1 处画一条横线
    plt.axhline(y=1, color='r', linestyle='--', label='R=1')

    # 设置 y 轴范围从 0 开始
    plt.ylim(0, 1.1)
    plt.xlim(0, index + 1)  # 设置 x 轴范围
    plt.title("Scatter Plot of R Values for Different Jitter Percent")
    plt.xlabel("R Value Index (Order)")
    plt.ylabel("R = max_reaction_time / final_e2e_max")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(f"{R_plot_name}")  # 保存图像
    
    print(f"R plot saved to {R_plot_name}")




if __name__ == "__main__":
    main()