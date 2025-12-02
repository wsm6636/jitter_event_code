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

import copy
import math
import random
import numpy as np
from scipy.optimize import basinhopping

from analysis_passive import  Task, RandomEvent

from analysis_active import combine_with_insertion, chain_asc_no_free_jitter_active, our_chain_active


# chain: sort by desc index
def chain_desc_no_free_jitter_active(tasks):
    n = len(tasks)
    current_task = tasks[-1]
    all_bridges = []
    adjusted = False

    for i in range(n-2, -1,-1):  
        result = combine_with_insertion(current_task, tasks[i])
        if result is False:
            return False
        else:
            r, w, adjusted, bridges = result
            current_task = Task(read_event=r, write_event=w, id=r.id)
            if bridges:
                all_bridges.append((i, bridges))
    return r, w, adjusted, all_bridges, current_task



#chain max period order
def chain_max_period_active(tasks):
    max_period_index, max_period_task = max(
    enumerate(tasks),
    key=lambda t: t[1].period
)

    all_bridges = []
    pred_bridges = []
    succ_bridges = []
    adjusted = False

    #Grouping
    predecessor_group = tasks[:max_period_index + 1]  #task0~i
    successor_group = tasks[max_period_index + 1:]       #taski+1~n

    #task0~i chain
    if len(predecessor_group) > 1:
        predecessor_result = chain_desc_no_free_jitter_active(predecessor_group)
        if predecessor_result is not False:
            _, _, adjusted, pred_bridges, predecessor_task = predecessor_result
        else:
            return 0, None, None, adjusted, all_bridges
    elif len(predecessor_group) == 1:
        predecessor_task = predecessor_group[0]

    if successor_group:
        successor_group.insert(0, predecessor_task)

    if successor_group:
        successor_result = chain_asc_no_free_jitter_active(successor_group)
        if successor_result is not False:
            final_r, final_w, adjusted_succ, succ_bridges_temp, final_task = successor_result
            for idx, bridge_list in succ_bridges_temp:
                global_idx = idx + max_period_index+1
                succ_bridges.append((global_idx, bridge_list))

            adjusted = adjusted or adjusted_succ
            all_bridges = pred_bridges + succ_bridges
            max_reaction_time = final_w.offset + final_w.maxjitter - final_r.offset + final_r.period
            return max_reaction_time, final_r, final_w, adjusted, all_bridges
        else:
            return 0, None, None, adjusted, all_bridges
    else:
        all_bridges = pred_bridges
        max_reaction_time = predecessor_task.write_event.offset + predecessor_task.write_event.maxjitter - predecessor_task.read_event.offset + predecessor_task.read_event.period
        return max_reaction_time, predecessor_task.read_event, predecessor_task.write_event, adjusted, all_bridges


#chain min period order
def chain_min_period_active(tasks):
    min_period_index, min_period_task = min(
    enumerate(tasks),
    key=lambda t: (t[1].period, t[0])
)
    all_bridges = []
    pred_bridges = []
    succ_bridges = []
    adjusted = False

    #Grouping
    predecessor_group = tasks[:min_period_index + 1]  #task0~i
    successor_group = tasks[min_period_index + 1:]       #taski+1~n

    #task0~i chain
    if len(predecessor_group) > 1:
        predecessor_result = chain_desc_no_free_jitter_active(predecessor_group)
        if predecessor_result is not False:
            _, _, adjusted, pred_bridges, predecessor_task = predecessor_result
        else:
            return 0, None, None, adjusted, all_bridges
    elif len(predecessor_group) == 1:
        predecessor_task = predecessor_group[0]

    if successor_group:
        successor_group.insert(0, predecessor_task)
        
    if successor_group:
        successor_result = chain_asc_no_free_jitter_active(successor_group)
        if successor_result is not False:
            final_r, final_w, adjusted_succ, succ_bridges_temp, final_task = successor_result
            for idx, bridge_list in succ_bridges_temp:
                global_idx = idx + min_period_index+1
                succ_bridges.append((global_idx, bridge_list))
            adjusted = adjusted or adjusted_succ
            all_bridges = pred_bridges + succ_bridges
            max_reaction_time = final_w.offset + final_w.maxjitter - final_r.offset + final_r.period
            return max_reaction_time, final_r, final_w, adjusted, all_bridges
        else:
            return 0, None, None, adjusted, all_bridges
    else:
        all_bridges = pred_bridges
        max_reaction_time = predecessor_task.write_event.offset + predecessor_task.write_event.maxjitter - predecessor_task.read_event.offset + predecessor_task.read_event.period
        return max_reaction_time, predecessor_task.read_event, predecessor_task.write_event, adjusted, all_bridges


def our_chain_desc_active(tasks):
    if len(tasks) == 1:
        # Our analysis is also valid for a task
        final_r = tasks[0].read_event
        final_w = tasks[0].write_event
        max_reaction_time = final_w.offset + final_w.maxjitter - final_r.offset + final_r.period
        return max_reaction_time, final_r, final_w, False
    else:
        final_combine_result = chain_desc_no_free_jitter_active(tasks)
    if final_combine_result:
        final_r, final_w, adjusted, all_bridges, _ = final_combine_result
        # max reaction time need to add the period of the first read event
        max_reaction_time = final_w.offset + final_w.maxjitter - final_r.offset + final_r.period
        return max_reaction_time, final_r, final_w, adjusted, all_bridges
    else:
        return False
    


def inject_bridges(tasks, all_bridges):
    """
    Processing the adjusted task set.
    Insert the bridge tasks into the original task set.
    arguments:
        tasks: the original task set
        all_bridges: the list of all bridge tasks and their positions
    return:
        tasks: the adjusted task set with bridge tasks inserted
    """
    for idx, bridge_list in reversed(all_bridges):
        for b in reversed(bridge_list):
            tasks.insert(idx, b)
    return tasks



def find_valid_task_chains(tasks):
    """
    Generate a general task chain.
    Satisfy the read and write time order.
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
    Calculate the reaction time of the general task chain.
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
    Objective function for optimization.
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
    check if the new solution is within bounds.
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
    Maximize the reaction time of the general task chain.
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


def run_analysis_active_our_order(num_tasks, periods,read_offsets,write_offsets, per_jitter, chain_types):
    global results_function
    results_function = []  
    inserted = False
    adjusted = False

    tasks = RandomEvent(num_tasks, periods,read_offsets,write_offsets, per_jitter).tasks

    # final = our_chain_active(tasks)
    # Define the available chain functions
    available_chain_functions = {
        'asc': our_chain_active,
        'desc': our_chain_desc_active,
        'max_period': chain_max_period_active,
        'min_period': chain_min_period_active,
    }

    # Create chain_functions dictionary based on the provided chain_types
    chain_functions = {chain_type: available_chain_functions[chain_type] for chain_type in chain_types}       

    results = {}
    mrt_results = {}
    
    for name, func in chain_functions.items():
        print(f"Running chain type: {name}")    
        
        result = func(tasks)

        new_tasks = tasks

        if result:
            final_e2e_max, final_r, final_w, adjusted, all_bridges = result
            if all_bridges:
                new_tasks = inject_bridges(tasks[:], all_bridges)
                inserted = True
                
            results[name] = (final_e2e_max, final_r, final_w, adjusted, inserted)
        else:
            results[name] = (0, None, None, adjusted, inserted)
        
        reaction_time_a = maximize_reaction_time(new_tasks)
        reaction_time_b = max(results_function)
        max_reaction_time = max(reaction_time_a, reaction_time_b)
        mrt_results[name] = (max_reaction_time)
    
    return results, mrt_results, new_tasks


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

    final = our_chain_active(tasks)
    final_e2e_max = final[0]
    
    print(final_e2e_max)


