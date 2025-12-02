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
from analysis_passive import euclide_extend
from periodic_LET import combine_zero_jitter


def adjust_offsets(read_offset, write_offset, period, write_jitter, read_jitter):
    """
    When Eq.(18) cannot be satisfied, adjust the read offset of a pair (w, r).
    For more information, please see Corollary (2) in the paper.
    arguments:
        read_offset: read offset of the read event
        write_offset: write offset of the write event
        period: period of the events
        write_jitter: max jitter of the write event
        read_jitter: max jitter of the read event
    return:
        read_offset: adjusted read offset of the read event
        ad_scuss: whether the adjustment is successful
    """
    ad_scuss = None
    old_read_offset = read_offset
    delta_mod_period = (read_offset - write_offset) % period
    # If the original offset satisfies Theorem (2), return directly
    if write_jitter <= delta_mod_period and delta_mod_period < (period - read_jitter):    
        return read_offset, ad_scuss
    else:
        print(f"------------------adjust read offsets-----------------")

    # Traverse all candidate read offsets
    r_offsets = []
    step=0.1
    for current_read_offset in np.arange(0, period, step):
        delta_mod_period = (current_read_offset - write_offset) % period  # Calculate the difference modulo period
        if write_jitter <= delta_mod_period < (period - read_jitter):
            r_offsets.append(current_read_offset)

    if r_offsets:
        # Select a new offset
        read_offset_random = random.choice(r_offsets)
        valid_offsets = [x for x in r_offsets if x < old_read_offset]
        if valid_offsets:
            read_offset_abs = min(valid_offsets, key=lambda x: abs(x - old_read_offset))
        else:
            read_offset_abs = random.choice(r_offsets)
        # Use the smaller value of the two to ensure conservatism
        read_offset = min(read_offset_abs, read_offset_random)
        ad_scuss = True
        return read_offset, ad_scuss
    else:
        ad_scuss = False
        print("No valid offsets found within the given constraints.")
        return read_offset, ad_scuss
    


def effective_event_active(task1, task2):
    """
    Find effective event with adjustment of offset.
    Please see line 4, 8, 12 in Algorithm (2) in the paper.
    arguments:
        task1: the task τi which write data
        task2: the task τi+1 which read data
    return:
        (w_star, r_star, adjusted): the effective events and whether adjustment is successful
        False: if no effective events can be found
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
    Combine tasks in the chain with adjustment of offset.
    Please see Algorithm (2) in the paper.
    arguments:
        task1: the task τi which write data
        task2: the task τi+1 which read data
    return:
        (r_1_2, w_1_2, adjusted): the combined read and write events and whether adjustment is successful
        False: if no combined events can be found
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
    Please see Section VI "Active analysis" in the paper.
    arguments:
        task1: the task τi which write data
        task2: the task τi+1 which read data
    return:
        (r_1_2, w_1_2, adjusted, bridges): the combined read and write events, whether adjustment is successful, and the inserted bridge tasks
        False: if no combined events can be found
    """
    adjusted = False
    direct = combine_no_free_jitter_active(task1, task2)
    if direct is not False:
        r, w, adjusted = direct          
        return r, w, adjusted, []  # no insertion needed, return the combined task
    elif task1.period == task2.period:
        return False  # cannot combine tasks with the same period without insertion
    else:
        # Generate bridge task
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
        # Recombining bridge tasks and chains
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



def combine_zero_jitter(task1, task2):
    T1 = task1.period
    rd_ph1 = task1.read_offset + task1.read_event.maxjitter
    wr_ph1 = task1.write_offset + task1.write_event.maxjitter
    T2 = task2.period
    rd_ph2 = task2.read_offset + task2.read_event.maxjitter
    wr_ph2 = task2.write_offset + task2.write_event.maxjitter

    # distance from tau_1 job 0 write  -->  tau_2 job 0 read
    PPhase = rd_ph2-wr_ph1

    # G = GCD(T1,T2)
    # c1, c2 are coefficients of Bezout's identity: c1*T1+c2*T2 = G
    (G,c1,c2) = euclide_extend(T1,T2)
    p1 = T1//G
    p2 = T2//G

    # Minimum latency
    min_latency = (wr_ph1-rd_ph1)+(wr_ph2-rd_ph2)+(PPhase % G)

    # Parameters of the 1 -> 2 chain
    if T1 == T2:
        # assuming jobs of the chain 1->2 are indexed by T1
        rd_ph12 = rd_ph1
        rd_delta12 = wr_delta12 = T1
        cycle = 1
        wr_ph12 = wr_ph2-PPhase+(PPhase % T1)
        max_latency = min_latency
        id_min_latency = 0
        id_max_latency = 0
        return None, None

    elif T1 > T2:
        T12 = T1
        phi1 = (PPhase % T2) // G      # Eq. (23)
        rd_ph12 = rd_ph1
        rd_delta12 = T1                # constant separation of consecutive reads
        dancing = [(phi1-j1*p1) % p2 for j1 in range(p2)]  # the job-dependent piece of (22)
        cycle = p2
        # Write phasing, Eq. (24)
        wr_ph12 = [wr_ph2-PPhase+rem*G+(PPhase % G) for rem in dancing]
        # separation of consecutive writes
        wr_delta12 = [(p1//p2)*T2 if (rem >= p1 % p2) else (p1//p2+1)*T2 for rem in dancing]
        inv_p1 = c1 % p2   # multiplicative inverse of p1 over modulo-p2
        # Notice that min/max below are computed without enumerating dancing
        max_latency = min_latency+T2-G
        id_min_latency = (phi1*inv_p1) % p2      # same id as min in dancing
        id_max_latency = ((phi1+1)*inv_p1) % p2  # same id as max in dancing
        # Read phase of the copier task after tau_2. Eq. (37)
        rd_ph2next = wr_ph2-PPhase+(PPhase % G)+T2-G

        copier_id = f"{task2.id}_next_copier"
        copier_offset = rd_ph2next     

    else:
        T12 = T2
        phi2 = (PPhase % T1) // G      # Eq. (32)
        wr_ph12 = wr_ph2
        wr_delta12 = T2                # constant separation of consecutive writes
        dancing = [(phi2+j2*p2) % p1 for j2 in range(p1)]  # the job-dependent piece of (31)
        cycle = p1
        # Read phasing, Eq. (33)
        rd_ph12 = [rd_ph1+PPhase-(PPhase % G)-rem*G for rem in dancing]
        # separation of consecutive reads
        rd_delta12 = [(p2//p1+1)*T1 if (rem >= (-p2) % p1) else (p2//p1)*T1 for rem in dancing]
        inv_p2 = c2 % p1   # multiplicative inverse of p2 over modulo-p1
        # Notice that min/max below are computed without enumerating dancing
        max_latency = min_latency+T1-G
        id_min_latency = (-phi2*inv_p2) % p1     # same id as max in dancing
        id_max_latency = (-(phi2+1)*inv_p2) % p1 # same id as min in dancing
        # Write phase of the copier task before tau_1. Eq. (40)
        wr_ph1prev = rd_ph1+PPhase-(PPhase % G)-T1+G

        copier_id = f"{task1.id}_prev_copier"
        copier_offset = wr_ph1prev


    add_r = Event(
        id=f"{copier_id}_read",
        event_type="read_copier",
        period=T12,
        offset=copier_offset,
        maxjitter=0,
    )
    add_w = Event(
        id=f"{copier_id}_write",
        event_type="write_copier",
        period=T12,
        offset=copier_offset,
        maxjitter=0,
    )

    return add_r, add_w

def chain_asc_no_free_jitter_active(tasks):
    """
    Processing Chain with adjustment of offset.
    In ascending order of the task's index in the chain.
    arguments:
        tasks: the task set
    return:
        (r_final, w_final, adjusted, all_bridges): the combined read and write events, whether adjustment is successful, and the inserted bridge tasks
        False: if no combined events can be found
    """
    n = len(tasks)
    # Start from the head of the chain and combine backwards
    current_task = tasks[0]
    all_bridges = []
    adjusted = False

    for i in range(1, n):
        result = combine_with_insertion(current_task, tasks[i])

        # if result is False:
        #     return False
        # else:
        #     r, w, adjusted, bridges = result
        #     current_task = Task(read_event=r, write_event=w, id=r.id)
        #     if bridges:
        #         all_bridges.append((i, bridges))

        if result is False:
            # Fallback: try zero-jitter combination (no bridge, no offset change)
            result_zj = combine_zero_jitter(current_task, tasks[i])
            if result_zj is False:
                # Even zero-jitter failed → entire chain fails
                return False
            else:
                r, w = result_zj
                # No bridges inserted, no adjustment in this step
                if r is None or w is None:
                    return False
                else:
                    current_task = Task(read_event=r, write_event=w, id=r.id)
                    all_bridges.append((i, current_task))
                # Note: we do NOT mark adjusted=True here
        else:
            r, w, adjusted, bridges = result
            current_task = Task(read_event=r, write_event=w, id=r.id)
            if bridges:
                all_bridges.append((i, bridges))

    return  r, w, adjusted, all_bridges



def our_chain_active(tasks):
    """
    The maximum reaction time results DFF_bound in our paper Formula (39).
    arguments:
        tasks: the task set
    return:
        max_reaction_time: the maximum reaction time of the chain
        final_r: the final read event of the chain
        final_w: the final write event of the chain
        adjusted: whether adjustment is successful
        all_bridges: the list of all bridge tasks and their positions
        False: if no valid task chain can be found
    """
    if len(tasks) == 1:
        # Our analysis is also valid for a task
        final_r = tasks[0].read_event
        final_w = tasks[0].write_event
        max_reaction_time = final_w.offset + final_w.maxjitter - final_r.offset + final_r.period
        return max_reaction_time, final_r, final_w
    else:
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


def run_analysis_active_our(num_tasks, periods,read_offsets,write_offsets, per_jitter):
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
        new_tasks: the adjusted task set with bridge tasks inserted
        adjusted: whether adjustment is successful
        inserted: whether bridge tasks are inserted
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
    arguments:
        num_tasks: number of tasks in the chain
        periods: list of periods of the tasks
        read_offsets: list of read offsets of the tasks
        write_offsets: list of write offsets of the tasks
        per_jitter: jitter percentage (for period)
    return:
        final_e2e_max: the maximum reaction time of our active analysis
        final_r: the final read event of the chain
        final_w: the final write event of the chain
        new_tasks: the adjusted task set with bridge tasks inserted
        adjusted: whether adjustment is successful
        inserted: whether bridge tasks are inserted
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
    arguments:
        num_tasks: number of tasks in the chain
        periods: list of periods of the tasks       
        read_offsets: list of read offsets of the tasks
        write_offsets: list of write offsets of the tasks   
        read_jitters: list of read jitters of the tasks
        write_jitters: list of write jitters of the tasks
    return:
        final_e2e_max: the maximum reaction time of our active analysis
        final_r: the final read event of the chain
        final_w: the final write event of the chain
        new_tasks: the adjusted task set with bridge tasks inserted
        adjusted: whether adjustment is successful  
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


