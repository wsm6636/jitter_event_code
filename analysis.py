#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 05 10:25:52 2025

It implements the methods described in the paper
   Shumo Wang, Enrico Bini, Martina Maggio
   "Understanding Jitter Propagation in Task Chains"

@author: Shumo Wang
"""

import math
import random
import numpy as np
from scipy.optimize import basinhopping


class Event:
    def __init__(self, event_type, period, offset, maxjitter, id=None):
        self.id = id
        self.event_type = event_type  # "read" or "write"
        self.period = period
        self.offset = offset
        self.maxjitter = maxjitter
        self.random_jitter = 0   # jitter of instance

    def __repr__(self):
        return (
            f"Event(type={self.event_type},id={self.id}, period={self.period}, "
            f"offset={self.offset}, maxjitter={self.maxjitter}"
        )

    def get_trigger_time(self, j):
        self.random_jitter = random.uniform(0, self.maxjitter)
        tj = j * self.period + self.offset + self.random_jitter
        return tj


class Task:
    def __init__(self, read_event, write_event, id=None):
        self.id = id
        self.read_event = read_event
        self.write_event = write_event
        self.period = read_event.period
        self.offset = read_event.offset

    def __repr__(self):
        return (
            f"Task(period={self.period}, offset={self.offset}, "
            f"read_event={self.read_event}, write_event={self.write_event})"
        )


# random events generator
class RandomEvent:
    def __init__(
        self,
        num_tasks,
        periods,
        read_offsets,
        write_offsets,
        per_jitter
    ):
        self.num_tasks = num_tasks
        self.periods = periods
        self.read_offsets = read_offsets
        self.write_offsets = write_offsets
        self.per_jitter = per_jitter
        self.tasks = self.generate_events_tasks()
        
    def generate_events_tasks(self):
        read_events = []
        write_events = []
        events = []
        tasks = []
        for i in range(self.num_tasks):
            # randomly select a period from the list
            period = self.periods[i]
            read_offset = self.read_offsets[i]
            write_offset = self.write_offsets[i]
            # x% * period
            maxjitter = self.per_jitter*period

            # create read and write events
            read_event = Event(
                event_type="read",
                period=period,
                offset=read_offset,
                maxjitter=maxjitter,
                id=i,
            )
            write_event = Event(
                event_type="write",
                period=period,
                offset=write_offset,
                maxjitter=maxjitter,
                id=i,
            )
            read_events.append(read_event)
            write_events.append(write_event)
            events.append((read_event, write_event))

            # Create a task with the read and write events
            task = Task(read_event=read_event, write_event=write_event, id=i)
            tasks.append(task)

            # print(f"task {i}: read_event: period: {read_event.period}, offset: {read_event.offset}, maxjitter: {read_event.maxjitter}")
            # print(f"task {i}: write_event: period: {write_event.period}, offset: {write_event.offset}, maxjitter: {write_event.maxjitter}")
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


# find effective event
# Algorithm 2 line 1
def effective_event(w, r):
    w_star = None
    r_star = None
    delta = r.offset - w.offset

    (G, pw, pr) = euclide_extend(w.period, r.period)
    T_star = max(w.period, r.period)

    if w.period == r.period:  # Theorem 2
        if (w.maxjitter <= (delta % T_star) and (delta % T_star) < (T_star - r.maxjitter)):  # Formula (16)
            w_jitter_star = w.maxjitter
            r_jitter_star = r.maxjitter  # Formula (17)
            if delta < 0:
                w_offser_star = w.offset
                r_offset_star = w.offset + (delta % T_star)  # Formula (17)
            else:
                w_offser_star = r.offset - (delta % T_star)  # Formula (17)
                r_offset_star = r.offset
        else:
            print(f"Does not conform to Theorem 2, Formula (16).")
            return False
    elif w.period > r.period:
        if (r.period + r.maxjitter) <= (w.period - w.maxjitter):  # Formula (19) Theorem (3)
            kw = max(0, (((delta + r.maxjitter - r.period) // w.period) + 1))  # Formula (19)
            w_offser_star = w.offset + kw * w.period
            w_jitter_star = w.maxjitter
            r_offset_star = w_offser_star
            r_jitter_star = r.period + w.maxjitter  # Formula (20)
        else:
            print(f"Does not conform to Theorem (13), Formula (17).")
            return False
    elif w.period < r.period:
        if (w.period + w.maxjitter) <= (r.period - r.maxjitter):  # Formula (24) Theorem (4)
            kr = max(0, math.ceil((w.maxjitter - delta) / r.period))  # Formula (25)
            r_offset_star = r.offset + kr * r.period
            r_jitter_star = r.maxjitter  # Formula (25)
            w_offser_star = r_offset_star - w.period
            w_jitter_star = w.period + r.maxjitter  # Formula (26)
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
        maxjitter=w_jitter_star,
    )
    r_star = Event(
        id=r.id,
        event_type="read_star",
        period=T_star,
        offset=r_offset_star,
        maxjitter=r_jitter_star,
    )
    # (w_star, r_star) the reslut of effective event (line 1)
    return (w_star, r_star)


# Algorithm 2
def combine_no_free_jitter(task1, task2):
    r1 = task1.read_event
    w1 = task1.write_event
    r2 = task2.read_event
    w2 = task2.write_event
    # line 1
    result = effective_event(w1, r2)

    if result:
        (w1_star, r2_star) = result
    else:
        # print("==========FAILED TO EFFECTIVE EVENT==========")
        return False
    
    T_star = w1_star.period  # line 2
    if task1.period > task2.period:  # line 4
        r_1_2_offset = r1.offset + w1_star.offset - w1.offset  # line 5
        r_1_2_jitter = r1.maxjitter  # line 6
        m2 = w2.offset - r2.offset - r2.maxjitter
        M2 = w2.offset - r2.offset + w2.maxjitter  # line 7
        w_1_2_offset = r2_star.offset + m2
        w_1_2_jitter = r2_star.maxjitter + M2 - m2  # line 8
    elif task1.period < task2.period:  # line 9
        w_1_2_offset = w2.offset + r2_star.offset - r2.offset  # line 10
        w_1_2_jitter = w2.maxjitter  # line 11
        m1 = w1.offset - r1.offset - r1.maxjitter
        M1 = w1.offset - r1.offset + w1.maxjitter  # line 12
        r_1_2_offset = w1_star.offset - M1
        r_1_2_jitter = w1_star.maxjitter + M1 - m1  # line 13
    else:  # line 14
        r_1_2_offset = r1.offset + w1_star.offset - w1.offset
        r_1_2_jitter = r1.maxjitter
        w_1_2_offset = w2.offset + r2_star.offset - r2.offset
        w_1_2_jitter = w2.maxjitter

    combined_id = f"{task1.id}_{task2.id}"
    r_1_2 = Event(
        id=combined_id,
        event_type="read_combined",
        period=T_star,
        offset=r_1_2_offset,
        maxjitter=r_1_2_jitter,
    )  # line 19
    w_1_2 = Event(
        id=combined_id,
        event_type="write_combined",
        period=T_star,
        offset=w_1_2_offset,
        maxjitter=w_1_2_jitter,
    )  # line 20

    return (r_1_2, w_1_2)

# asc index order chain 
def chain_asc_no_free_jitter(tasks):
    n = len(tasks)
    current_task = tasks[0]

    for i in range(1, n):
        result = combine_no_free_jitter(current_task, tasks[i])
        if result is False:
            # print(f"Failed to combine task {current_task.id} and task {tasks[i].id}.")
            return False
        else:
            (r, w) = result
            current_task = Task(read_event=r, write_event=w, id=r.id)
    # if r.offset < 0:
    #   print(f"r.offset < 0. r.offset: {r.offset:.2f}, w.offset: {w.offset:.2f}.")
    #   w.offset -= r.offset
    #   r.offset = 0
    #   r.offset += r.period
    #   w.offset += r.period
    return  r, w

# max reaction time of our paper
def our_chain(tasks):
    final_combine_result = chain_asc_no_free_jitter(tasks)
    if final_combine_result:
        final_r, final_w = final_combine_result
        # max reaction time need to add the period of the first read event
        max_reaction_time = final_w.offset + final_w.maxjitter - final_r.offset + final_r.period
        return max_reaction_time, final_r, final_w
    else:
        # print("Failed to combine predecessor and successor results.")
        return False

# general task chain
# Satisfy the order of read and write times
def find_valid_task_chains(tasks):
    task_chain = []
    last_write_time = -float("inf")
    for task in tasks:
        read_event = task.read_event
        write_event = task.write_event
        
        if task is tasks[0]:
            read_instance = objective_function.iteration
        else:
            read_instance = 0

        # find the first read instance that satisfies the condition
        while True:
            read_time = read_event.get_trigger_time(read_instance)
            if read_time >= last_write_time:
                break
            read_instance += 1

        write_time = write_event.get_trigger_time(read_instance)

        read_event.read_time = read_time
        write_event.write_time = write_time

        task_chain.append((read_event, read_time, read_instance))
        task_chain.append((write_event, write_time, read_instance))

        last_write_time = write_time

    # check if the task chain is valid
    if len(task_chain) == len(tasks) * 2:
        return task_chain
    else:
        # print("Invalid task chain generated.")
        return False


# Calculate the reaction time of the general task chain
# also need to add the period of the first read event
def calculate_reaction_time(task_chain):
    first_read_time = task_chain[0][1]
    last_write_time = task_chain[-1][1]
    reaction_time = last_write_time - first_read_time +  task_chain[0][0].period

    return reaction_time  


# Objective function for optimization
def objective_function(x, tasks):
    num_tasks = len(tasks)
    for i in range(num_tasks):
        tasks[i].read_event.random_jitter = x[i]
        tasks[i].write_event.random_jitter = x[i + num_tasks]

    task_chain = find_valid_task_chains(tasks)

    if task_chain:
        max_reaction_time = calculate_reaction_time(task_chain)

        objective_function.iteration += 1
        results_function.append(max_reaction_time)
        return -max_reaction_time
    else:
        return float("inf")
    
# Iteration
def take_step(x, bounds):
    new_x = x.copy()
    for i in range(len(x)):
        lower, upper = bounds[i]
        new_x[i] = random.uniform(lower, upper)
    # print(f"take_step: {new_x}")
    return new_x

# check if the new solution is within bounds
def accept_test(f_new, x_new, f_old, x_old, tasks, bounds, **kwargs):
    for i, (lower, upper) in enumerate(bounds):
        if not (lower <= x_new[i] <= upper):
            return False
    return True
    
# Maximize the reaction time of the general task chain
def maximize_reaction_time(tasks):
    bounds = [(0, 0)] * (len(tasks) * 2)
    initial_guess = [0] * len(tasks) * 2
    for i, task in enumerate(tasks):
        bounds[i] = (0, task.read_event.maxjitter)
        bounds[i+ len(tasks)] = (0, task.write_event.maxjitter)
        # guess the initial value : random jitter of read and write events
        initial_guess[i] = random.uniform(0, task.read_event.maxjitter)
        initial_guess[i + len(tasks)] = random.uniform(0, task.write_event.maxjitter)

    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}

    objective_function.iteration = 0

    def objective(x):
        return objective_function(x, tasks)
    def accept(f_new, x_new, f_old, x_old, **kwargs):
        return accept_test(f_new, x_new, f_old, x_old, tasks, bounds, **kwargs)

    # Use basinhopping to find the global maximum reaction time
    result = basinhopping(
        objective,
        initial_guess,
        minimizer_kwargs=minimizer_kwargs,
        niter=1,
        T=1.0,
        stepsize=1.0,  # Step size for the random walk
        interval=50,  # Interval for the random walk
        niter_success=10,  # Iteration bound
        # stepsize=0.01,
        take_step=lambda x: take_step(x, bounds),
        accept_test=accept
    )
    max_reaction_time = -result.fun
    
    return max_reaction_time

def davare_reaction_time(tasks):
    davare_e2e = 0
    for task in tasks:
        task_rt = task.write_event.offset + task.write_event.maxjitter - task.read_event.offset
        davare_e2e += task.period + task_rt
    return davare_e2e

def duerr_reaction_time(tasks):
    last_task = tasks[-1]
    first_task = tasks[0]
    last_task_rt = last_task.write_event.offset + last_task.write_event.maxjitter - last_task.read_event.offset

    duerr_e2e = last_task_rt + first_task.period
    for (task, next_task) in zip(tasks[:-1], tasks[1:]):
        task_rt = task.write_event.offset + task.write_event.maxjitter - task.read_event.offset
        duerr_e2e += max(task_rt, next_task.period + task_rt)

    return duerr_e2e

results_function = []

# outport function
def run_analysis(num_tasks, periods,read_offsets,write_offsets, per_jitter):
    global results_function
    results_function = []  

    tasks = RandomEvent(num_tasks, periods,read_offsets,write_offsets, per_jitter).tasks

    final = our_chain(tasks)
    
    if final is False:
        final_e2e_max = 0
        final_r = None
        final_w = None
    else:
        final_e2e_max = final[0]
        final_r = final[1]
        final_w = final[2]
        
    # check if the final result is valid
    reaction_time_a = maximize_reaction_time(tasks)
    reaction_time_b = max(results_function)
    max_reaction_time = max(reaction_time_a, reaction_time_b)
    # max_reaction_time = 0

    # davare_e2e = davare_reaction_time(tasks)
    # duerr_e2e = duerr_reaction_time(tasks)

    return final_e2e_max, max_reaction_time, final_r, final_w, tasks


# test the code
if __name__ == "__main__":
    num_tasks = 5 
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000]
    per_jitter = 0.05 # percent jitter
    read_offsets = [0, 0, 0, 0, 0]
    write_offsets = [1, 1, 1, 1, 1]

    run_analysis(num_tasks, periods,read_offsets,write_offsets, per_jitter)


