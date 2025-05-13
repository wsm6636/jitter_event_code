#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 05 10:25:52 2025

It implements the methods described in the paper
   Shumo Wang, Enrico Bini, Martina Maggio
   "Understanding Jitter Propagation in Task Chains"

@author: Shumo Wang
"""

# Event and Task classes
import datetime
import math
import random
import numpy as np
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt
import seaborn as sns

class Event:
    def __init__(self, event_type, period, offset, jitter, id=None):
        self.id = id
        self.event_type = event_type  # "read" or "write"
        self.period = period
        self.offset = offset
        self.jitter = jitter
        self.read_time = 0
        self.write_time = 0
        # print(f"event {self.event_type}_{self.id},  period: {self.period}, offset: {self.offset}, jitter: {self.jitter}.")

    def __repr__(self):
        return (
            f"Event(type={self.event_type},id={self.id}, period={self.period}, "
            f"offset={self.offset}, jitter={self.jitter}, read_time={self.read_time:.2f}, "
            f"write_time={self.write_time:.2f}"
        )

    def get_trigger_time(self, j):
        random_jitter = random.uniform(0, self.jitter)
        tj = j * self.period + self.offset + random_jitter
        # print(f"event {self.event_type}_{self.id}, j: {j}, trigger_time: {tj:.2f}.")
        return tj


class Task:
    def __init__(self, read_event, write_event, id=None):
        self.id = id
        self.read_event = read_event
        self.write_event = write_event
        self.period = read_event.period
        self.offset = read_event.offset
        self.jitter = 0
        # print(f"task_{self.id}, period: {self.period}, offset: {self.offset}, read_event: {self.read_event.event_type}_{self.read_event.id}, write_event: {self.write_event.event_type}_{self.write_event.id}.")

    def __repr__(self):
        return (
            f"Task(period={self.period}, offset={self.offset}, "
            f"read_event={self.read_event}, write_event={self.write_event})"
        )


# random events
class RandomEvent:
    def __init__(
        self,
        num_tasks,
        periods,
        max_offset,
        max_jitter,
    ):
        self.num_tasks = num_tasks
        self.periods = periods
        self.max_offset = max_offset
        self.max_jitter = max_jitter
        self.tasks = self.generate_events_tasks()

    def generate_events_tasks(self):
        read_events = []
        write_events = []
        events = []
        tasks = []
        for i in range(self.num_tasks):
            # 随机生成周期
            period = random.choice(self.periods)

            # 随机生成偏移量，确保偏移量小于周期
            read_offset = random.randint(
                0, min(self.max_offset, period - 1)
            )
            # LET
            write_offset = read_offset + period

            # 随机生成抖动
            jitter = random.randint(0, self.max_jitter)

            # 创建读事件和写事件
            read_event = Event(
                event_type="read",
                period=period,
                offset=read_offset,
                jitter=jitter,
                id=i,
            )
            write_event = Event(
                event_type="write",
                period=period,
                offset=write_offset,
                jitter=jitter,
                id=i,
            )
            read_events.append(read_event)
            write_events.append(write_event)
            events.append((read_event, write_event))

            task = Task(read_event=read_event, write_event=write_event, id=i)
            tasks.append(task)

        return tasks

    def get_tasks(self):
        return self.tasks



# Euclide's algorithm for coefficients of Bezout's identity
def euclide_extend(a, b):
    r0 = int(a)
    r1 = int(b)
    s0 = 1
    s1 = 0
    t0 = 0
    t1 = 1
    while r1 != 0:
        q = r0 // r1
        new_r = r0 % r1
        new_s = s0 - q * s1
        new_t = t0 - q * t1
        r0 = r1
        s0 = s1
        t0 = t1
        r1 = new_r
        s1 = new_s
        t1 = new_t
    return (r0, s0, t0)


# effective event
# Algorithm 2 line 1
def effective_event(w, r):
    w_star = None
    r_star = None
    delta = r.offset - w.offset
    #    print(f"delta: {delta}.")
    (G, pw, pr) = euclide_extend(w.period, r.period)
    # print(f"G: {G}.")
    T_star = max(w.period, r.period)
    #    print(f"T_star: {T_star}.")

    if w.period == r.period:  # Theorem 12
        #   print(f"periods are equal. Theorem 12.")
        if (
            w.jitter <= (delta % T_star) & (delta % T_star) < (T_star - r.jitter)
        ):  # Formula (14)
            w_jitter_star = w.jitter
            r_jitter_star = r.jitter  # Formula (15)
            if delta < 0:
                # print(f"delta < 0. Formula (15).")
                w_offser_star = w.offset
                r_offset_star = w.offset + (delta % T_star)  # Formula (15)
            else:
                # print(f"delta >= 0. Formula (15).")
                w_offser_star = r.offset - (delta % T_star)  # Formula (15)
                r_offset_star = r.offset
        else:
            # print(f"Does not conform to Theorem 12, Formula (14).")
            return False
    elif w.period > r.period:
        if w.jitter == r.jitter == 0:  # Lemma (15)
            #  print(f"w.period > r.period, without jitter. Lemma (15), Formula (28).")
            kw = max(0, ((delta - r.period) // w.period) + 1)
            w_offser_star = w.offset + kw * w.period
            w_jitter_star = 0
            r_offset_star = w_offser_star + (delta % G)
            r_jitter_star = r.period - G  # Formula (28)
        elif (r.period + r.jitter) <= (
            w.period - w.jitter
        ):  # Formula (17) Theorem (13)
            #  print(f"w.period > r.period, with jitter. Theorem (13), Formula (17).")
            kw = max(0, ((delta + r.jitter - r.period) // w.period) + 1)  # Formula (19)
            w_offser_star = w.offset + kw * w.period
            w_jitter_star = w.jitter
            r_offset_star = w_offser_star
            r_jitter_star = r.period + w.jitter  # Formula (18)
        else:
            # print(f"Does not conform to Theorem (13), Formula (17).")
            return False
    elif w.period < r.period:
        if w.jitter == r.jitter == 0:  # Lemma (16)
            #  print(f"w.period < r.period, without jitter. Lemma (16), Formula (30).")
            kr = max(0, math.ceil(-delta / r.period))
            r_offset_star = r.offset + kr * r.period
            r_jitter_star = 0
            w_offser_star = r_offset_star - (delta % G) - w.period + G
            w_jitter_star = w.period - G  # Formula (30)
        elif (w.period + w.jitter) <= (
            r.period - r.jitter
        ):  # Formula (22) Theorem (14)
            #  print(f"w.period < r.period, with jitter. Theorem (14), Formula (22).")
            kr = max(0, math.ceil((w.jitter - delta) / r.period))  # Formula (25)
            r_offset_star = r.offset + kr * r.period
            r_jitter_star = r.jitter  # Formula (23)
            w_offser_star = r_offset_star - w.period
            w_jitter_star = w.period + r.jitter  # Formula (24)
        else:
            # print(f"Does not conform to Theorem (14), Formula (22).")
            return False
    else:
        # print(f"Does not exist effective write/read event series.")
        return False

    w_star = Event(
        id=w.id,
        event_type="write_star",
        period=T_star,
        offset=w_offser_star,
        jitter=w_jitter_star,
    )
    r_star = Event(
        id=r.id,
        event_type="read_star",
        period=T_star,
        offset=r_offset_star,
        jitter=r_jitter_star,
    )
    # print(f"w_star : period: {w_star.period}, offset: {w_star.offset}, jitter: {w_star.jitter}")
    # print(f"r_star : period: {r_star.period}, offset: {r_star.offset}, jitter: {r_star.jitter}")

    return (w_star, r_star)


def combine_no_free_jitter(task1, task2):
    r1 = task1.read_event
    w1 = task1.write_event
    r2 = task2.read_event
    w2 = task2.write_event

    result = effective_event(w1, r2)
    # print(result)
    if result:
        (w1_star, r2_star) = result
    else:
        # print("==========FAILED TO EFFECTIVE EVENT==========")
        return False
    T_star = w1_star.period  # line 2
    if task1.period > task2.period:  # line 4
        r_1_2_offset = r1.offset + w1_star.offset - w1.offset  # line 5
        r_1_2_jitter = r1.jitter  # line 6
        m2 = w2.offset - r2.offset - r2.jitter
        M2 = w2.offset - r2.offset + w2.jitter  # line 7
        w_1_2_offset = r2_star.offset + m2
        w_1_2_jitter = r2_star.jitter + M2 - m2  # line 8
    elif task1.period < task2.period:  # line 9
        w_1_2_offset = w2.offset + r2_star.offset - r2.offset  # line 10
        w_1_2_jitter = w2.jitter  # line 11
        m1 = w1.offset - r1.offset - r1.jitter
        M1 = w1.offset - r1.offset + w1.jitter  # line 12
        r_1_2_offset = w1_star.offset - M1
        r_1_2_jitter = w1_star.jitter + M1 - m1  # line 13
    else:  # line 14
        r_1_2_offset = r1.offset + w1_star.offset - w1.offset
        r_1_2_jitter = r1.jitter
        w_1_2_offset = w2.offset + r2_star.offset - r2.offset
        w_1_2_jitter = w2.jitter

    combined_id = f"{task1.id}_{task2.id}"
    r_1_2 = Event(
        id=combined_id,
        event_type="read_combined",
        period=T_star,
        offset=r_1_2_offset,
        jitter=r_1_2_jitter,
    )  # line 19
    w_1_2 = Event(
        id=combined_id,
        event_type="write_combined",
        period=T_star,
        offset=w_1_2_offset,
        jitter=w_1_2_jitter,
    )  # line 20
    # print(f"period: {r_1_2.period}, offset: {r_1_2.offset}, jitter: {r_1_2.jitter}")
    # print(f"period: {w_1_2.period}, offset: {w_1_2.offset}, jitter: {w_1_2.jitter}")
    return (r_1_2, w_1_2)


# e2e
def e2e(r, w):
    #    print("================E2E====================")
    min_e2e = w.offset - r.offset - r.jitter
    max_e2e = w.offset + w.jitter - r.offset
    #    print(f"min_e2e: {min_e2e}, max_e2e: {max_e2e}")
    return (min_e2e, max_e2e)


def chain_asc_no_free_jitter(tasks):
    #    print("================CHAIN_ASC====================")
    n = len(tasks)
    current_task = tasks[0]

    for i in range(1, n):
        #   print(f"================Combining task {current_task.id} and {tasks[i].id}====================")
        result = combine_no_free_jitter(current_task, tasks[i])
        if result is False:
            #  print("================CHAIN_ASC END====================")
            # print(f"Failed to combine task {current_task.id} and task {tasks[i].id}.")
            return False
        else:
            (r, w) = result
            #  print("================UPDATE combined task====================")
            current_task = Task(read_event=r, write_event=w, id=r.id)

    return e2e(r, w), r, w, current_task


def our_chain(tasks):
    final_combine_result = chain_asc_no_free_jitter(tasks)
    if final_combine_result:
        final_e2e, final_r, final_w, final_task = final_combine_result
        # print(
        #     f"final_e2e: min_e2e: {final_e2e[0]}, max_e2e: {final_e2e[1]}, max_reaction_time: {final_e2e[1] + final_r.period}, min_reaction_time: {final_e2e[0] + final_r.period}"
        # )
        # print(
        #     f"final_r: period: {final_r.period}, offset: {final_r.offset}, jitter: {final_r.jitter}"
        # )
        # print(
        #     f"final_w: period: {final_w.period}, offset: {final_w.offset}, jitter: {final_w.jitter}"
        # )
        return final_e2e, final_r, final_w, final_task
    else:
        #   print("Failed to combine predecessor and successor results.")
        return False


    

def objective_function2(x, num_tasks, periods, max_offset, max_jitter,visited_results, iteration):
    tasks = []
    for i in range(num_tasks):
        period = int(x[i * 3])
        offset = int(x[i * 3 + 1])
        jitter = int(x[i * 3 + 2])

        read_event = Event(event_type="read", period=period, offset=offset, jitter=jitter, id=i)
        write_event = Event(event_type="write", period=period, offset=offset+period, jitter=jitter, id=i)
        tasks.append(Task(read_event=read_event, write_event=write_event, id=i))

    our = our_chain(tasks)
    
    if our is False:
        return 999
    else:
        final_e2e, final_r, final_w, final_task = our
        final_e2e_max = final_e2e[1] + final_r.period

    # 检查是否重复
    if final_e2e_max in visited_results:
        with open(log_file2, "a") as file:
            file.write(f"检测到超周期！迭代次数: {iteration}, max_reaction_time2: {final_e2e_max}\n")
        print(f"检测到超周期！迭代次数: {iteration}, max_reaction_time2: {final_e2e_max}")
    else:
        visited_results.add(final_e2e_max)

    if log2 is True:
        with open(log_file2, "a") as file:
            file.write(f"==================Iteration AG: {objective_function2.iteration}======================\n")
            if our is not False:
                file.write(f"final_e2e: max_reaction_time: {final_e2e_max}\n")
                file.write(f"final_r: period: {final_r.period}, offset: {final_r.offset}, jitter: {final_r.jitter}\n")
                file.write(f"final_w: period: {final_w.period}, offset: {final_w.offset}, jitter: {final_w.jitter}\n")
                file.write(f"Tasks:\n")
                for i, task in enumerate(tasks):
                    file.write(f"   Task {i}: {task.read_event.event_type}_{task.read_event.id}, "
                            f"period={task.read_event.period}, offset={task.read_event.offset}, jitter={task.read_event.jitter}\n")
                    file.write(f"   Task {i}: {task.write_event.event_type}_{task.write_event.id}, "
                            f"period={task.write_event.period}, offset={task.write_event.offset}, jitter={task.write_event.jitter}\n")
            else:
                file.write(f"NULL\n")

    objective_function2.iteration += 1
    results_function2.append(final_e2e_max)

    return -final_e2e_max

def take_step(x):
    new_x = x.copy()
    i = random.randint(0, len(x) // 3 - 1)
    while True:
        new_period = random.choice(periods)  # 从指定的周期列表中随机选择周期
        if new_period in periods:
            break
    # print(f"new_period: {new_period}")
    new_x[3*i] = new_period  # 更新周期值
    new_x[3*i + 1] = random.randint(0, max_offset)  # 随机改变偏移
    new_x[3*i + 2] = random.randint(0, max_jitter)  # 随机改变抖动


    return new_x


def maximize_reaction_time(num_tasks, periods, max_offset, max_jitter):
    initial_guess = []
    for _ in range(num_tasks):
        initial_guess.extend([
            random.choice(periods),  # period
            random.randint(0, max_offset),           # offset
            random.randint(0, max_jitter)            # jitter
        ])
    print(f"Initial guess: {initial_guess}")
    bounds = [(periods[0], periods[-1]), (0, max_offset), (0, max_jitter)] * num_tasks

    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
    # 初始化记录状态的集合
    visited_results = set()
    # 使用闭包封装 objective_function

    def objective2(x):
        return objective_function2(x, num_tasks, periods, max_offset, max_jitter, visited_results, objective_function2.iteration)


    objective_function2.iteration = 0


    result2 = basinhopping(
        objective2,
        initial_guess,
        minimizer_kwargs=minimizer_kwargs,
        niter=2000,
        T=1.0,
        take_step=lambda x: take_step(x),
        accept_test=lambda x_new, x_old, **kwargs: True
    )   

    max_reaction_time2 = -result2.fun
    return  max_reaction_time2

def plot_reaction_time_distribution( results2, title="Max Reaction Time Distribution", fig_file=None):
    """
    绘制两个目标函数的最大反应时间分布的箱形图，并显示最大值、最小值和平均值。

    参数:
    results1 (list): 第一个目标函数的结果列表。
    results2 (list): 第二个目标函数的结果列表。
    title (str): 图表的标题，默认为 "Max Reaction Time Distribution"。
    """

    # 将结果转换为适合绘图的格式
    data = {
        "AG2": results2 if results2 else [0]
    }

    if not results2:
        AG2 = {"max": 0, "min": 0, "mean": 0}
    else:
        AG2 = {
            "max": np.max(results2),
            "min": np.min(results2),
            "mean": np.mean(results2)
        }
    # 创建箱形图
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data)
    plt.title(title)
    plt.ylabel("Max Reaction Time")
    plt.xlabel(f"num_tasks {num_tasks}\n"
        f"AG2 (Max: {AG2['max']:.2f}, Min: {AG2['min']:.2f}, Mean: {AG2['mean']:.2f})")


    # 添加标注
    stats = [AG2]  # 将两个目标函数的统计信息存储在一个列表中
    for i, values in enumerate(stats):
        max_val = values["max"]
        min_val = values["min"]
        mean_val = values["mean"]
        
        # 添加最大值标注
        plt.text(i, max_val, f"Max: {max_val:.2f}", ha="center", va="bottom", fontsize=10, color="blue")
        
        # 添加最小值标注
        plt.text(i, min_val, f"Min: {min_val:.2f}", ha="center", va="top", fontsize=10, color="red")
        
        # 添加平均值标注
        plt.text(i, mean_val, f"Mean: {mean_val:.2f}", ha="center", va="bottom", fontsize=10, color="black")

    plt.savefig(fig_file)
    plt.show()

def plot_scatter( results2, title="Max Reaction Time Scatter Plot", fig_file=None):
    """
    绘制两个目标函数的最大反应时间的散点图。

    参数:
    results1 (list): 第一个目标函数的结果列表。
    results2 (list): 第二个目标函数的结果列表。
    title (str): 图表的标题，默认为 "Max Reaction Time Scatter Plot"。
    fig_file (str): 保存图表的文件路径。
    """
    plt.figure(figsize=(10, 6))
    # plt.scatter(range(len(results2)), results2, label="AG2", color="red", alpha=0.6)
    plt.plot(range(len(results2)), results2, label="AG2", color="red", marker='x', linestyle='-', linewidth=1, markersize=4)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Max Reaction Time")
    plt.legend()
    if fig_file:
        plt.savefig(fig_file)
    plt.show()

# init
print("================INIT====================")

num_tasks = 5  # 任务数量
periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000]
# periods = [1, 2, 5, 10, 20]
max_offset = 10  # 最大偏移量
max_jitter = 5  # 最大抖动
all_final_e2e_max = []

results_function2 = []
log2 = False


print(f"num_tasks: {num_tasks}, max_offset: {max_offset}, max_jitter: {max_jitter}")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

log_file2 = f"AG_{num_tasks}_{timestamp}.txt"
fig_file = f"box_{num_tasks}_{timestamp}.png"

print("================REACTION TIME ANALYSIS====================")
max_reaction_time2 = maximize_reaction_time(num_tasks, periods, max_offset, max_jitter)
print(f"AG Maximized reaction time: {max_reaction_time2:.2f}")

if log2 is True:
    with open(log_file2, "a") as file:  
        file.write(f"==================Final AG======================\n")
        file.write(f"AG Maximized reaction time: {max_reaction_time2:.2f}\n")
    print(f"Results written to {log_file2}")

print("================PLOTTING====================")
# 绘制箱形图
# plot_reaction_time_distribution( results_function2, title="Max Reaction Time Distribution", fig_file=fig_file)

scatter_fig_file = f"scatter_{num_tasks}_{timestamp}.png"
plot_scatter(results_function2, title="Max Reaction Time Scatter Plot", fig_file=scatter_fig_file)