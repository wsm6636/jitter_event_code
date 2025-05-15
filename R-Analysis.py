# 文件名：main.py
import datetime
from WATER_period import run_analysis
import numpy as np

def main():
    num_repeats = 5  # 重复次数
    num_tasks = 5  # 任务数量
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000]  # 周期列表
    niter = 1  # 迭代次数
    per_jitter = 0.05

    results = []
    final = []
    false_results = []


    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"result/{num_repeats}_{num_tasks}_{niter}_{timestamp}.txt"

    for i in range(num_repeats):
        print(f"================== Repeat {i} ==================")
        final_e2e_max, max_reaction_time,  final_r, final_w, tasks = run_analysis(num_tasks, periods, per_jitter, niter)
        if final_e2e_max != 0:
            r = max_reaction_time / final_e2e_max
        else:
            r = None
        results.append((final_e2e_max, max_reaction_time,r))
        final.append((final_r, final_w))

    # 计算 final_e2e_max = 0 的百分比
    final_e2e_max_values = [result[0] for result in results]
    zero_count = final_e2e_max_values.count(0)
    zero_percentage = (zero_count / num_repeats) * 100
    false_results.append(zero_count)


    # 打印结果
    print("\n================== Results ==================")
    for i, (final_e2e_max, max_reaction_time, r) in enumerate(results):
        if r is not None:
            print(f"Repeat {i}: Final E2E Max: {final_e2e_max:.2f}, Maximized Reaction Time: {max_reaction_time:.2f}, R = {r:.2f}")
        else:
            print(f"Repeat {i}: Final E2E Max: {final_e2e_max:.2f}, Maximized Reaction Time: {max_reaction_time:.2f}, R = None")

    print(f"Percentage of False: {zero_percentage:.2f}%")



    # 打印最终结果
    with open(file_name, "a") as file:
        for i, (final_e2e_max, max_reaction_time, r) in enumerate(results):
            file.write(f"================== Repeat {i} ==================\n")
            for task in tasks:
                file.write(f"task_{i}: read_event: {task.read_event.event_type}_{task.read_event.id}, "
                        f"period: {task.read_event.period}, offset: {task.read_event.offset}, maxjitter: {task.read_event.maxjitter};\n")
                file.write(f"task_{i}: write_event: {task.write_event.event_type}_{task.write_event.id}, "
                        f"period: {task.write_event.period}, offset: {task.write_event.offset}, maxjitter: {task.write_event.maxjitter}.\n")
                
            file.write(f"Final Results\n")

            if final_e2e_max != 0:
                file.write(f"   Final R: period:{final_r.period}, offset:{final_r.offset:.2f}, jitter:{final_r.maxjitter:.2f}\n")
                file.write(f"   Final W: period:{final_w.period}, offser:{final_w.offset:.2f}, jitter:{final_w.maxjitter:.2f}\n")
            else:
                file.write(f"   Final R and Final W None\n")

            file.write(f"   Final E2E Max: {final_e2e_max:.2f}\n")
            file.write(f"   Maximized Reaction Time: {max_reaction_time:.2f}\n")

            if r is not None:
                file.write(f"   R = max_reaction_time / final_e2e_max {r:.2f}\n")
            else:
                file.write(f"   R: None\n")

        file.write(f"   Percentage of False: {zero_percentage:.2f}%\n")

        print(f"Results saved to {file_name}")
            

if __name__ == "__main__":
    main()