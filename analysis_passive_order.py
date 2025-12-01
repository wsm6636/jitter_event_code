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

"""an event"""
class Event:
    """
    Creates an event represented by ID, type, period, offset, maxjitter.
    """
    def __init__(self, event_type, period, offset, maxjitter, id=None):
        self.id = id
        self.event_type = event_type  # "read" or "write"
        self.period = period
        self.offset = offset
        self.maxjitter = maxjitter  # J_i in our paper
        self.random_jitter = 0   # jitter of instance

    """print an event"""
    def __repr__(self):
        return (
            f"Event(type={self.event_type},id={self.id}, period={self.period}, "
            f"offset={self.offset}, maxjitter={self.maxjitter}"
        )

    """Generate an instance jitter"""
    def get_trigger_time(self, j):
        self.random_jitter = random.uniform(0, self.maxjitter)
        tj = j * self.period + self.offset + self.random_jitter
        return tj


"""a task"""
class Task:
    """
    Creates a task represented by ID, events, period, offset.
    """
    def __init__(self, read_event, write_event, id=None):
        self.id = id
        self.read_event = read_event
        self.write_event = write_event
        self.period = read_event.period
        self.offset = read_event.offset

    """print a task"""
    def __repr__(self):
        return (
            f"Task(period={self.period}, offset={self.offset}, "
            f"read_event={self.read_event}, write_event={self.write_event})"
        )


"""
Generate random events
"""
class RandomEvent:
    """
    Generates maximum jitter as percentage per_jitter 
    maxjitter = per_jitter * period
    """
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
        
    """
    Generating tasks with events
    """
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

        return tasks

    def get_tasks(self):
        return self.tasks
    

"""
Generate random events for Gunzel
"""
class RandomEventForGunzel:
    """
    Based on the set of tasks Gunzel generate, all jitters and offsets are known
    """
    def __init__(
        self,
        num_tasks,
        periods,
        read_offsets,
        write_offsets,
        read_jitters, 
        write_jitters
    ):
        self.num_tasks = num_tasks
        self.periods = periods
        self.read_offsets = read_offsets
        self.write_offsets = write_offsets
        self.read_jitters = read_jitters
        self.write_jitters = write_jitters
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
            read_maxjitter = self.read_jitters[i]
            write_maxjitter = self.write_jitters[i]

            # create read and write events
            read_event = Event(
                event_type="read",
                period=period,
                offset=read_offset,
                maxjitter=read_maxjitter,
                id=i,
            )
            write_event = Event(
                event_type="write",
                period=period,
                offset=write_offset,
                maxjitter=write_maxjitter,
                id=i,
            )
            read_events.append(read_event)
            write_events.append(write_event)
            events.append((read_event, write_event))

            # Create a task with the read and write events
            task = Task(read_event=read_event, write_event=write_event, id=i)
            tasks.append(task)

        return tasks

    def get_tasks(self):
        return self.tasks



def euclide_extend(a, b):
    """
    Euclide's algorithm for coefficients of Bezout's identity
    """
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



def effective_event(w, r):
    """
    Find effective event without adjustment of offset
    Please see line 4, 8, 12 in Algorithm (2) in the paper.
    arguments:
        w: write event
        r: read event   
    return: 
        (w_star, r_star): the effective events after combination
        False: if the events do not conform to the theorems
    """
    w_star = None
    r_star = None
    delta = r.offset - w.offset

    (G, pw, pr) = euclide_extend(w.period, r.period)
    T_star = max(w.period, r.period)  # Algorithm(2) line2

    if w.period == r.period:  # Algorithm(2) line 11, Theorem(2) 
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
    elif w.period < r.period: # Algorithm(2) line 7, Theorem(4)
        if (w.period + w.maxjitter) <= (r.period - r.maxjitter):  # Formula (29) 
            # Formula (30)
            kr = max(0, math.ceil((w.maxjitter - delta) / r.period))  
            r_offset_star = r.offset + kr * r.period
            r_jitter_star = r.maxjitter  # 
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
    return (w_star, r_star)



def combine_no_free_jitter(task1, task2):
    """
    Combine tasks in the chain without adjustment of offset
    Please see Algorithm (2) in the paper.
    arguments:
        task1: the task τi which write data
        task2: the task τi+1 which read data
    return:
        (r_1_2, w_1_2): the combined read and write events of the two tasks
        False: if the events do not conform to the theorems
    """
    r1 = task1.read_event
    w1 = task1.write_event
    r2 = task2.read_event
    w2 = task2.write_event
    
    result = effective_event(w1, r2)

    if result:
        (w1_star, r2_star) = result
    else:
        return False
    
    T_star = w1_star.period  # line 2
    if task1.period > task2.period:  # line 3
        r_1_2_offset = r1.offset + w1_star.offset - w1.offset  # line 5, Eq.(34)
        r_1_2_jitter = r1.maxjitter  
        m2 = w2.offset - r2.offset - r2.maxjitter   # Eq.(35)
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
        r_1_2_offset = r1.offset + w1_star.offset - w1.offset # line 13, Eq.(34)
        r_1_2_jitter = r1.maxjitter
        w_1_2_offset = w2.offset + r2_star.offset - r2.offset # line 14, Eq.(37)
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
    return (r_1_2, w_1_2)




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
    max_period_task = min(tasks, key=lambda x: (x.period, -tasks.index(x)))
    max_period_index = tasks.index(max_period_task)

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
    min_period_task = min(tasks, key=lambda x: (x.period, tasks.index(x)))
    min_period_index = tasks.index(min_period_task)

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


