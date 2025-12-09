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
import numpy as np
from scipy.optimize import basinhopping

from analysis_passive import Event, Task, RandomEvent, RandomEventForGunzel
from analysis_passive import combine_no_free_jitter


def combine_with_insertion_add(task1, task2):
    """
    Eliminating jitter with a “prefix” event series r`i and a “postfix” series w`i (Formula (40))
    For the write task τi and the read task τi+1, if the conditions of formula (23) or (29) do not hold.
    Please see Section VI "Active analysis" in the paper.
    arguments:
        task1: the task τi which write data
        task2: the task τi+1 which read data
    return:
        (r_1_2, w_1_2): the combined read and write events, whether adjustment is successful, and the inserted bridge tasks
        False: if no combined events can be found
    """
    added = False
    direct = combine_no_free_jitter(task1, task2)
    if direct is not False:
        r, w = direct          
        return r, w, added
    else:
        # Generate bridge task
        w1  = task1.write_event
        r2  = task2.read_event

        w1_prime = Event(
            id=f"{w1.id}_prime",
            event_type="write_prime",
            period=w1.period,
            offset=w1.offset+w1.maxjitter,
            maxjitter=0,
        )
        r2_prime = Event(
            id=f"{r2.id}_prime",
            event_type="read_prime",
            period=r2.period,
            offset=r2.offset,
            maxjitter=0,
        )

        # Reprocess the parts with prefix and suffix
        # Recombining bridge tasks and chains
        task1.write_event = w1_prime
        step1 = combine_no_free_jitter(task1, task2)
        if step1 is False:
            task1.write_event = w1  # restore
            task2.read_event = r2_prime
            step2 = combine_no_free_jitter(task1, task2)
            if step2 is False:
                task1.write_event = w1_prime
                task2.read_event = r2_prime
                step3 = combine_no_free_jitter(task1, task2)
                if step3 is False:
                    task1.write_event = w1  # restore
                    task2.read_event = r2  # restore
                    return False
                else:
                    added = True
                    r, w = step3
            else:
                added = True
                r, w = step2
        else:
            added = True   
            r, w = step1
        
    return r, w, added
            




def chain_asc_no_free_jitter_active_add(tasks):
    """
    Processing Chain with adjustment of offset.
    In ascending order of the task's index in the chain.
    arguments:
        tasks: the task set
    return:
        (r_final, w_final, added): the combined read and write events, whether adjustment is successful, and the inserted bridge tasks
        False: if no combined events can be found
    """
    n = len(tasks)
    # Start from the head of the chain and combine backwards
    current_task = tasks[0]
    added = False

    for i in range(1, n):
        result = combine_with_insertion_add(current_task, tasks[i])
        if result is False:
            return False
        else:
            r, w, added = result
            current_task = Task(read_event=r, write_event=w, id=r.id)

    return  r, w, added



def our_chain_active_add(tasks):
    """
    The maximum reaction time results DFF_bound in our paper Formula (39).
    arguments:
        tasks: the task set
    return:
        max_reaction_time: the maximum reaction time of the chain
        final_r: the final read event of the chain
        final_w: the final write event of the chain
        added: whether adjustment is successful
        False: if no valid task chain can be found
    """
    if len(tasks) == 1:
        # Our analysis is also valid for a task
        final_r = tasks[0].read_event
        final_w = tasks[0].write_event
        max_reaction_time = final_w.offset + final_w.maxjitter - final_r.offset + final_r.period
        return max_reaction_time, final_r, final_w
    else:
        final_combine_result = chain_asc_no_free_jitter_active_add(tasks)
    if final_combine_result is False:
        return False
    else:
        final_r, final_w, added = final_combine_result
        # max reaction time need to add the period of the first read event
        max_reaction_time = final_w.offset + final_w.maxjitter - final_r.offset + final_r.period
        return max_reaction_time, final_r, final_w, added


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


def run_analysis_active_our_add(num_tasks, periods,read_offsets,write_offsets, per_jitter):
    """
    Generate a task set based on random parameters (IC/LET jitter=0)
    Generate a general task chain
    Calculate the maximum reaction time between us and the general task chain (our active analysis)
    arguments:
        num_tasks: number of tasks in the chain
        periods: list of periods of the tasks
        read_offsets: list of read offsets of the tasks
        write_offsets: list of write offsets of the tasks
        per_jitter: jitter percentage (for period)
    return:
        final_e2e_max: the maximum reaction time of our active analysis
        max_reaction_time: the maximum reaction time of the general task chain
        final_r: the final read event of the chain
        final_w: the final write event of the chain
        new_tasks: the added task set with bridge tasks inserted
        added: whether adjustment is successful
        inserted: whether bridge tasks are inserted
    """
    global results_function
    results_function = []  

    tasks = RandomEvent(num_tasks, periods,read_offsets,write_offsets, per_jitter).tasks

    final = our_chain_active_add(tasks)
    
    new_tasks = tasks
    if final is False:
        final_e2e_max = 0
        final_r = None
        final_w = None
        added = False
    else:
        final_e2e_max = final[0]
        final_r = final[1]
        final_w = final[2]
        added = final[3]


    # check if the final result is valid
    reaction_time_a = maximize_reaction_time(new_tasks)
    reaction_time_b = max(results_function)
    max_reaction_time = max(reaction_time_a, reaction_time_b)

    return final_e2e_max, max_reaction_time, final_r, final_w, new_tasks, added



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

    final = our_chain_active_add(tasks)
    final_e2e_max = final[0]
    
    print(final_e2e_max)


