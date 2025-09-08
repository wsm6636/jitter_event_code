#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 05 10:25:52 2025

It implements the methods described in the paper
    Shumo Wang, Enrico Bini, Martina Maggio, Qingxu Deng
    "Jitter in Task Chains"

@author: Shumo Wang
"""

import math
import random
import numpy as np
from scipy.optimize import basinhopping

from analysis_passive import Event, Task, RandomEvent, RandomEventForGunzel
from analysis_passive import euclide_extend



def adjust_offsets(read_offset, write_offset, period, write_jitter, read_jitter):
    """
    Corollary (2)
    When Eq.(18) cannot be satisfied, adjust the read offset of a pair (w, r)
    """
    ad_scuss = None
    old_read_offset = read_offset
    delta_mod_period = (read_offset - write_offset) % period
    if write_jitter <= delta_mod_period and delta_mod_period < (period - read_jitter):    
        return read_offset, ad_scuss
    else:
        print(f"------------------adjust read offsets-----------------")

    r_offsets = []
    step=0.1
    for current_read_offset in np.arange(0, period, step):
        delta_mod_period = (current_read_offset - write_offset) % period  # Calculate the difference modulo period
        if write_jitter <= delta_mod_period < (period - read_jitter):
            r_offsets.append(current_read_offset)

    if r_offsets:
        read_offset = min(r_offsets, key=lambda x: abs(x - old_read_offset))
        ad_scuss = True
        return read_offset, ad_scuss
    else:
        ad_scuss = False
        print("No valid offsets found within the given constraints.")
        return read_offset, ad_scuss
    


def effective_event_active(task1, task2):
    """
    Find effective event with adjustment of offset
    line 4, 8, 12 in Algorithm (2) 
    """
    r1 = task1.read_event
    w = task1.write_event
    r = task2.read_event
    w2 = task2.write_event

    adjusted = False

    r_of_old = r.offset
    w2_of_old = w2.offset

    w_star = None
    r_star = None
    delta = r.offset - w.offset

    (G, pw, pr) = euclide_extend(w.period, r.period)
    T_star = max(w.period, r.period) # Algorithm(2) line2

    if w.period == r.period:  # Algorithm(2) line 11, Theorem(2) 
        # Try adjusting
        r_of_new, adjusted = adjust_offsets(r.offset, w.offset, T_star, w.maxjitter, r.maxjitter)
        delta = r_of_new - w.offset  # Update delta after adjustment
        if adjusted:
            # update events
            w2.offset = r_of_new - r_of_old + w2_of_old
            r.offset = r_of_new
            task2.read_event = r
            task2.write_event = w2
            print(f"Adjusted offsets: w.id {w.id}, with r.id {r.id}. r_of_old {r_of_old}, r.offset{r.offset}. w2_of_old {w2_of_old}, w2.offset {w2.offset}.")

        if (w.maxjitter <= (delta % T_star) and (delta % T_star) < (T_star - r.maxjitter)):  # Formula (18)
            # Formula (19)
            w_jitter_star = w.maxjitter
            r_jitter_star = r.maxjitter   
            if delta < 0:
                w_offser_star = w.offset
                r_offset_star = w.offset + (delta % T_star)  
            else:
                w_offser_star = r.offset - (delta % T_star)   
                r_offset_star = r.offset
        else:
            print(f"Does not conform to Theorem (2), Formula (18).")
            return False
    elif w.period > r.period: # Algorithm(2) line 3, Theorem(3)
        if (r.period + r.maxjitter) <= (w.period - w.maxjitter):  # Formula (23)
            # Formula (24)
            kw = max(0, (((delta + r.maxjitter - r.period) // w.period) + 1)) 
            w_offser_star = w.offset + kw * w.period
            w_jitter_star = w.maxjitter
            r_offset_star = w_offser_star
            r_jitter_star = r.period + w.maxjitter   
        else:
            print(f"Does not conform to Theorem (3), Formula (23).")
            return False
    elif w.period < r.period:  # Algorithm(2) line 7, Theorem(4)
        if (w.period + w.maxjitter) <= (r.period - r.maxjitter):  # Formula (29)
            # Formula (30)
            kr = max(0, math.ceil((w.maxjitter - delta) / r.period))   
            r_offset_star = r.offset + kr * r.period
            r_jitter_star = r.maxjitter   
            w_offser_star = r_offset_star - w.period
            w_jitter_star = w.period + r.maxjitter   
        else:
            print(f"Does not conform to Theorem (4), Formula (29).")
            return False
    else:
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
    # (w_star, r_star) the reslut of effective event (line 4, 8 ,12)
    return (w_star, r_star, adjusted)



def combine_no_free_jitter_active(task1, task2):
    """
    Combine tasks in the chain with adjustment of offset
    Algorithm(2)
    """
    r1 = task1.read_event
    w1 = task1.write_event
    r2 = task2.read_event
    w2 = task2.write_event

    result = effective_event_active(task1, task2)

    if result is False:
        return False
    else:
        w1_star, r2_star, adjusted = result

    T_star = w1_star.period  # line 2
    if task1.period > task2.period:  # line 3
        r_1_2_offset = r1.offset + w1_star.offset - w1.offset  # line 5, Eq.(34)
        r_1_2_jitter = r1.maxjitter   
        m2 = w2.offset - r2.offset - r2.maxjitter  # Eq.(35)
        M2 = w2.offset - r2.offset + w2.maxjitter   
        w_1_2_offset = r2_star.offset + m2          # line 6, Eq.(36)
        w_1_2_jitter = r2_star.maxjitter + M2 - m2   
    elif task1.period < task2.period:  # line 7
        w_1_2_offset = w2.offset + r2_star.offset - r2.offset  # line 9, Eq.(37)
        w_1_2_jitter = w2.maxjitter  # line 11
        m1 = w1.offset - r1.offset - r1.maxjitter  # Eq.(35)
        M1 = w1.offset - r1.offset + w1.maxjitter   
        r_1_2_offset = w1_star.offset - M1         # line 10, Eq.(38)
        r_1_2_jitter = w1_star.maxjitter + M1 - m1   
    else:  # line 11
        r_1_2_offset = r1.offset + w1_star.offset - w1.offset  # line 13, Eq.(34)
        r_1_2_jitter = r1.maxjitter
        w_1_2_offset = w2.offset + r2_star.offset - r2.offset  # line 14, Eq.(37)
        w_1_2_jitter = w2.maxjitter

    combined_id = f"{task1.id}_{task2.id}"
    r_1_2 = Event(
        id=combined_id,
        event_type="read_combined",
        period=T_star,
        offset=r_1_2_offset,
        maxjitter=r_1_2_jitter,
    )  # line 15
    w_1_2 = Event(
        id=combined_id,
        event_type="write_combined",
        period=T_star,
        offset=w_1_2_offset,
        maxjitter=w_1_2_jitter,
    )  # line 16

    # read and write series of the chain
    return (r_1_2, w_1_2, adjusted)



def combine_with_insertion(task1, task2):
    """
    Eliminating jitter with a “prefix” event series r`i and a “postfix” series w`i (Formula (40))
    For the write task τi and the read task τi+1, if the conditions of formula (23) or (29) do not hold.
    """
    adjusted = False
    direct = combine_no_free_jitter_active(task1, task2)
    if direct is not False:
        r, w, adjusted = direct          
        return r, w, adjusted, []  # no insertion needed, return the combined task
    elif task1.period == task2.period:
        return False  # cannot combine tasks with the same period without insertion
    else:
        
        w  = task1.write_event
        r  = task2.read_event

        w_bridge_task = Task(
            read_event=Event(id=f"{w.id}_bridge_r",
                            event_type="bridge_read",
                            period=w.period,
                            offset=w.offset + w.maxjitter,
                            maxjitter=0),
            write_event=Event(id=f"{w.id}_bridge_w",
                            event_type="bridge_write",
                            period=w.period,
                            offset=w.offset + w.maxjitter,
                            maxjitter=0),
            id=f"{w.id}_bridge"
        )

        r_bridge_task = Task(
            read_event=Event(id=f"{r.id}_bridge_r",
                            event_type="bridge_read",
                            period=r.period,
                            offset=r.offset,
                            maxjitter=0),
            write_event=Event(id=f"{r.id}_bridge_w",
                            event_type="bridge_write",
                            period=r.period,
                            offset=r.offset,
                            maxjitter=0),
            id=f"{r.id}_bridge"
        )
        
        # Reprocess the parts with prefix and suffix
        step1 = combine_no_free_jitter_active(task1, w_bridge_task)
        if not step1:
            return False
        r1, w1, adjusted = step1            

        step2 = combine_no_free_jitter_active(Task(read_event=r1, write_event=w1, id=w1.id),r_bridge_task)
        if not step2:
            return False
        r2, w2, adjusted = step2           

        step3 = combine_no_free_jitter_active(Task(read_event=r2, write_event=w2, id=w2.id),task2)
        if not step3:
            return False
        r_final, w_final,adjusted = step3

        bridges = [w_bridge_task, r_bridge_task]
        print(f"Bridges inserted: {w_bridge_task.id}, {r_bridge_task.id}")
        return r_final, w_final, adjusted, bridges


def chain_asc_no_free_jitter_active(tasks):
    """
    Processing Chain with adjustment of offset
    In ascending order of the task's index in the chain
    """
    n = len(tasks)
    current_task = tasks[0]
    all_bridges = []
    adjusted = False

    for i in range(1, n):
        result = combine_with_insertion(current_task, tasks[i])
        if result is False:
            return False
        else:
            r, w, adjusted, bridges = result
            current_task = Task(read_event=r, write_event=w, id=r.id)
            if bridges:
                all_bridges.append((i, bridges))
    return  r, w, adjusted, all_bridges



def our_chain_active(tasks):
    """
    The maximum reaction time results of our paper
    Formula (39) : DFF_bound
    """
    final_combine_result = chain_asc_no_free_jitter_active(tasks)
    if final_combine_result is False:
        return False
    else:
        final_r, final_w, adjusted, all_bridges = final_combine_result
        # max reaction time need to add the period of the first read event
        max_reaction_time = final_w.offset + final_w.maxjitter - final_r.offset + final_r.period
        return max_reaction_time, final_r, final_w, adjusted, all_bridges



def inject_bridges(tasks, all_bridges):
    """
    Processing the adjusted task set
    """
    for idx, bridge_list in reversed(all_bridges):
        for b in reversed(bridge_list):
            tasks.insert(idx, b)
    return tasks



def find_valid_task_chains(tasks):
    """
    Generate a general task chain
    Satisfy the read and write time order
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
    """
    first_read_time = task_chain[0][1]
    last_write_time = task_chain[-1][1]
    reaction_time = last_write_time - first_read_time +  task_chain[0][0].period
    
    return reaction_time



def objective_function(x, tasks):
    """
    The handle function of general task chain calculation
    Objective function for optimization
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
    Iteration
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
    """
    for i, (lower, upper) in enumerate(bounds):
        if not (lower <= x_new[i] <= upper):
            return False
    return True



def maximize_reaction_time(tasks):
    """
    Maximize the reaction time of the general task chain
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


def run_analysis_active_our(num_tasks, periods,read_offsets,write_offsets, per_jitter):
    """
    Generate a task set based on random parameters (IC/LET jitter=0)
    Generate a general task chain
    Calculate the maximum reaction time between us and the general task chain (our active analysis)
    """
    global results_function
    results_function = []  
    inserted = False

    tasks = RandomEvent(num_tasks, periods,read_offsets,write_offsets, per_jitter).tasks

    final = our_chain_active(tasks)
    
    new_tasks = tasks
    if final is False:
        final_e2e_max = 0
        final_r = None
        final_w = None
        adjusted = False
    else:
        final_e2e_max = final[0]
        final_r = final[1]
        final_w = final[2]
        adjusted = final[3]
        all_bridges = final[4]
        if all_bridges:
            # Processing the adjusted task set
            new_tasks = inject_bridges(tasks[:], all_bridges)
            inserted = True
            
    # check if the final result is valid
    reaction_time_a = maximize_reaction_time(new_tasks)
    reaction_time_b = max(results_function)
    max_reaction_time = max(reaction_time_a, reaction_time_b)

    return final_e2e_max, max_reaction_time, final_r, final_w, new_tasks, adjusted, inserted




def run_analysis_active_Gunzel_LET(num_tasks, periods,read_offsets,write_offsets, per_jitter):
    """
    Generate a LET task set based on random parameters
    Calculate the maximum reaction time (our active analysis)
    There is no need to compute a generic task chain, as the interface is used to compare the results of Gunzel LET (analysis_Gunzel.py/run_analysis_Gunzel_LET)
    """
    global results_function
    results_function = []  
    inserted = False

    tasks = RandomEvent(num_tasks, periods,read_offsets,write_offsets, per_jitter).tasks

    final = our_chain_active(tasks)
    
    new_tasks = tasks
    if final is False:
        final_e2e_max = 0
        final_r = None
        final_w = None
        adjusted = False
    else:
        final_e2e_max = final[0]
        final_r = final[1]
        final_w = final[2]
        adjusted = final[3]
        bridges = final[4]
        if bridges:
            new_tasks = inject_bridges(tasks[:], bridges)
            inserted = True
            
    return final_e2e_max, final_r, final_w, new_tasks, adjusted, inserted



def run_analysis_active_Gunzel_IC(num_tasks, periods,read_offsets,write_offsets, read_jitters, write_jitters ):
    """
    Generate a IC task set based on the known parameters obtained from Gunzel
    Calculate the maximum reaction time (our active analysis)
    There is no need to compute a generic task chain, as the interface is used to compare the results of Gunzel IC (analysis_Gunzel.py/run_analysis_Gunzel_IC)
    """
    global results_function
    results_function = []  
    inserted = False

    tasks = RandomEventForGunzel(num_tasks, periods,read_offsets,write_offsets, read_jitters, write_jitters, ).tasks

    final = our_chain_active(tasks)
    
    new_tasks = tasks

    if final is False:
        final_e2e_max = 0
        final_r = None
        final_w = None
        adjusted = False
    else:
        final_e2e_max = final[0]
        final_r = final[1]
        final_w = final[2]
        adjusted = final[3]
        bridges = final[4]
        if bridges:
            new_tasks = inject_bridges(tasks[:], bridges)
            inserted = True
            
    return final_e2e_max, final_r, final_w, new_tasks, adjusted, inserted


# test the code
if __name__ == "__main__":
    num_tasks = 5 
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000]
    per_jitter = 0.05 # percent jitter
    read_offsets = [0, 0, 0, 0, 0]
    write_offsets = [1, 1, 1, 1, 1]

    run_analysis_active_our(num_tasks, periods,read_offsets,write_offsets, per_jitter)


