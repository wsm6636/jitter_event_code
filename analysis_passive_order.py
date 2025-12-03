#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 05 10:25:52 2025

It implements the methods described in the paper
    "Jitter Propagation in Task Chains". 
    Shumo Wang, Enrico Bini, Qingxu Deng, Martina Maggio, 
    IEEE Real-Time Systems Symposium (RTSS), 2025

@author: Shumo Wang
"""

import math
import random
from scipy.optimize import basinhopping
from analysis_passive import  Task, RandomEvent, combine_no_free_jitter


def chain_asc_no_free_jitter(tasks):
    """
    Processing Chain without adjustment of offset
    In ascending order of the task's index in the chain
    arguments:
        tasks: the task set in the chain
    return: 
        (r, w): the combined read and write events of the chain
        False: if the events do not conform to the theorems
    """
    n = len(tasks)
    # Start from the head of the chain and combine backwards
    current_task = tasks[0]

    for i in range(1, n):
        result = combine_no_free_jitter(current_task, tasks[i])
        if result is False:
            return False
        else:
            (r, w) = result
            current_task = Task(read_event=r, write_event=w, id=r.id)
    return  r, w, current_task


# chain: sort by desc index
def chain_desc_no_free_jitter(tasks):
    n = len(tasks)
    current_task = tasks[-1]

    for i in range(n-2, -1,-1):  
        result = combine_no_free_jitter(tasks[i], current_task)
        if result is False:
            return False
        else:
            (r,w) = result
            current_task = Task(read_event=r, write_event=w, id=r.id)
    return r, w, current_task



#chain max period order
def chain_max_period(tasks):
    max_period_index, max_period_task = max(
    enumerate(tasks),
    key=lambda t: t[1].period
)
    #Grouping
    predecessor_group = tasks[:max_period_index + 1]  #task0~i
    successor_group = tasks[max_period_index + 1:]       #taski+1~n
    final_tasks = []

    #task0~i chain
    if len(predecessor_group) > 1:
        predecessor_result = chain_desc_no_free_jitter(predecessor_group)
        if predecessor_result is not False:
            _, _, predecessor_task = predecessor_result
        else:
            return 0, None, None
    elif len(predecessor_group) == 1:
        predecessor_task = predecessor_group[0]

    if successor_group:
        successor_group.insert(0, predecessor_task)

    if successor_group:
        successor_result = chain_asc_no_free_jitter(successor_group)
        if successor_result is not False:
            final_r, final_w, final_task = successor_result
            max_reaction_time = final_w.offset + final_w.maxjitter - final_r.offset + final_r.period
            return max_reaction_time, final_r, final_w
        else:
            return 0, None, None
    else:
        # 如果后继组为空，直接返回前驱组的结果
        max_reaction_time = predecessor_task.write_event.offset + predecessor_task.write_event.maxjitter - predecessor_task.read_event.offset + predecessor_task.read_event.period
        return max_reaction_time, predecessor_task.read_event, predecessor_task.write_event

#chain min period order
def chain_min_period(tasks):
    # 找到最小周期的任务，如果有多个，选择索引最小的那个
    min_period_index, min_period_task = min(
    enumerate(tasks),
    key=lambda t: (t[1].period, t[0])
)

    #Grouping
    predecessor_group = tasks[:min_period_index + 1]  #task0~i
    successor_group = tasks[min_period_index + 1:]       #taski+1~n
    final_tasks = []

    #task0~i chain
    if len(predecessor_group) > 1:
        predecessor_result = chain_desc_no_free_jitter(predecessor_group)
        if predecessor_result is not False:
            _, _, predecessor_task = predecessor_result
        else:
            return 0, None, None
    elif len(predecessor_group) == 1:
        predecessor_task = predecessor_group[0]

    if successor_group:
        successor_group.insert(0, predecessor_task)
        
    if successor_group:
        successor_result = chain_asc_no_free_jitter(successor_group)
        if successor_result is not False:
            final_r, final_w, final_task = successor_result
            max_reaction_time = final_w.offset + final_w.maxjitter - final_r.offset + final_r.period
            return max_reaction_time, final_r, final_w
        else:
            return 0, None, None
    else:
        # 如果后继组为空，直接返回前驱组的结果
        max_reaction_time = predecessor_task.write_event.offset + predecessor_task.write_event.maxjitter - predecessor_task.read_event.offset + predecessor_task.read_event.period
        return max_reaction_time, predecessor_task.read_event, predecessor_task.write_event

def our_chain(tasks):
    """
    The maximum reaction time results DFF_bound in our paper Formula (39).
    arguments:
        tasks: the task set in the chain
    return: 
        (max_reaction_time, r, w): the maximum reaction time and the combined read and write events of the chain
        False: if the events do not conform to the theorems
    """
    if len(tasks) == 1:
        # Our analysis is also valid for a task
        final_r = tasks[0].read_event
        final_w = tasks[0].write_event
        max_reaction_time = final_w.offset + final_w.maxjitter - final_r.offset + final_r.period
        return max_reaction_time, final_r, final_w
    else:
        final_combine_result = chain_asc_no_free_jitter(tasks)
    if final_combine_result:
        final_r, final_w, _ = final_combine_result
        # max reaction time need to add the period of the first read event
        max_reaction_time = final_w.offset + final_w.maxjitter - final_r.offset + final_r.period
        return max_reaction_time, final_r, final_w
    else:
        return False


def our_chain_desc(tasks):
    if len(tasks) == 1:
        # Our analysis is also valid for a task
        final_r = tasks[0].read_event
        final_w = tasks[0].write_event
        max_reaction_time = final_w.offset + final_w.maxjitter - final_r.offset + final_r.period
        return max_reaction_time, final_r, final_w
    else:
        final_combine_result = chain_desc_no_free_jitter(tasks)
    if final_combine_result:
        final_r, final_w, _ = final_combine_result
        # max reaction time need to add the period of the first read event
        max_reaction_time = final_w.offset + final_w.maxjitter - final_r.offset + final_r.period
        return max_reaction_time, final_r, final_w
    else:
        return False



def find_valid_task_chains(tasks):
    """
    Generate a general task chain
    Satisfy the read and write time order
    arguments:
        tasks: the task set
    return:
        task_chain: the valid task chain with read and write times
        False: if no valid task chain can be found
    """
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
        return False



def calculate_reaction_time(task_chain):
    """
    Calculate the reaction time of the general task chain
    arguments:
        task_chain: the valid task chain with read and write times
    return:
        reaction_time: the reaction time of the task chain
    """
    first_read_time = task_chain[0][1]
    last_write_time = task_chain[-1][1]
    reaction_time = last_write_time - first_read_time +  task_chain[0][0].period

    return reaction_time  



def objective_function(x, tasks):
    """
    The handle function of general task chain calculation
    Objective function for optimization
    arguments:
        x: the decision variable (jitter of read and write events)
        tasks: the task set
    return:
        -max_reaction_time: the negative of the maximum reaction time of the chain (for minimization)
        float("inf"): if no valid task chain can be found
    """
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
    

def take_step(x, bounds):
    """
    The handle function of general task chain calculation
    arguments:
        x: the decision variable (jitter of read and write events)
        bounds: the bounds of the decision variable
    return:
        new_x: the new decision variable after taking a step
    """
    new_x = x.copy()
    for i in range(len(x)):
        lower, upper = bounds[i]
        new_x[i] = random.uniform(lower, upper)
    return new_x


def accept_test(f_new, x_new, f_old, x_old, tasks, bounds, **kwargs):
    """
    The handle function of general task chain calculation
    check if the new solution is within bounds
    arguments:
        f_new: the new objective function value
        x_new: the new decision variable (jitter of read and write events)
        f_old: the old objective function value
        x_old: the old decision variable (jitter of read and write events)
        tasks: the task set
        bounds: the bounds of the decision variable
    return:
        True: if the new solution is within bounds
        False: if the new solution is out of bounds 
    """
    for i, (lower, upper) in enumerate(bounds):
        if not (lower <= x_new[i] <= upper):
            return False
    return True
    


def maximize_reaction_time(tasks):
    """
    Maximize the reaction time of the general task chain
    arguments:
        tasks: the task set
    return:         
        max_reaction_time: the maximum reaction time of the chain
    """
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

results_function = []


# outport function
def run_analysis_passive_our_order(num_tasks, periods,read_offsets,write_offsets, per_jitter, chain_types):
    global results_function
    results_function = []  

    tasks = RandomEvent(num_tasks, periods,read_offsets,write_offsets, per_jitter).tasks

    # Define the available chain functions
    available_chain_functions = {
        'asc': our_chain,
        'desc': our_chain_desc,
        'max_period': chain_max_period,
        'min_period': chain_min_period,
    }

    # Create chain_functions dictionary based on the provided chain_types
    chain_functions = {chain_type: available_chain_functions[chain_type] for chain_type in chain_types}

    results = {}
    for name, func in chain_functions.items():
        print(f"Running chain type: {name}")
        result = func(tasks)
        if result:
            max_reaction_time, final_r, final_w = result
            results[name] = (max_reaction_time, final_r, final_w)
        else:
            results[name] = (0, None, None)
        
    # check if the final result is valid
    reaction_time_a = maximize_reaction_time(tasks)
    reaction_time_b = max(results_function)
    max_reaction_time = max(reaction_time_a, reaction_time_b)
    # max_reaction_time = 0

    return results, max_reaction_time, tasks



# test the code
if __name__ == "__main__":
    num_tasks = 1 
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000]
    
    per_jitter = 0 # percent jitter

    selected_periods = [5]
    selected_read_offsets = [0]
    selected_write_offsets = [5]

    print(selected_periods)
    print(selected_read_offsets)
    print(selected_write_offsets)

    tasks = RandomEvent(num_tasks, selected_periods,selected_read_offsets,selected_write_offsets, per_jitter).tasks

    final = our_chain(tasks)
    final_e2e_max = final[0]
    
    print(final_e2e_max)


