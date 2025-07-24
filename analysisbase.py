#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 05 10:25:52 2025

It implements the methods described in the paper
   Shumo Wang, Enrico Bini, Martina Maggio
   "Understanding Jitter Propagation in Task Chains"

@author: Shumo Wang
"""

import datetime
import math
import random
import numpy as np
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt
import seaborn as sns



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
class RandomEvent_base:
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


results_function = []

# outport function
def run_analysis_base(num_tasks, periods,read_offsets,write_offsets, per_jitter):
    global results_function
    results_function = []  

    tasks = RandomEvent_base(num_tasks, periods,read_offsets,write_offsets, per_jitter).tasks
        
    # check if the final result is valid
    reaction_time_a = maximize_reaction_time(tasks)
    reaction_time_b = max(results_function)
    max_reaction_time = max(reaction_time_a, reaction_time_b)
    # max_reaction_time = 0
    return max_reaction_time, tasks


# test the code
if __name__ == "__main__":
    num_tasks = 5 
    periods = [1, 2, 5, 10, 20, 50, 100, 200, 1000]
    per_jitter = 0.05 # percent jitter
    read_offsets = [0, 0, 0, 0, 0]
    write_offsets = [1, 1, 1, 1, 1]

    run_analysis_base(num_tasks, periods,read_offsets,write_offsets, per_jitter)


