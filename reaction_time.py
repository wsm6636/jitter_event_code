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
import math
import numpy as np
from scipy.optimize import basinhopping


class Event:
    def __init__(self, event_type, period, offset, jitter,id=None):
        self.id = id
        self.event_type = event_type  # "read" or "write"
        self.period = period
        self.offset = offset
        self.jitter = jitter
        print(f"event {self.event_type}_{self.id},  period: {self.period}, offset: {self.offset}, jitter: {self.jitter}.")
    def __repr__(self):
        return (f"Event(type={self.event_type},id={self.id}, period={self.period}, "
            f"offset={self.offset}, jitter={self.jitter})")

class Task:
    def __init__(self, read_event, write_event, id=None):
        self.id = id
        self.read_event = read_event
        self.write_event = write_event
        self.period = read_event.period
        self.offset = read_event.offset
        self.jitter = 0
        print(f"task_{self.id}, period: {self.period}, offset: {self.offset}, read_event: {self.read_event.event_type}_{self.read_event.id}, write_event: {self.write_event.event_type}_{self.write_event.id}.")
    def __repr__(self):
        return (f"Task(period={self.period}, offset={self.offset}, "
            f"read_event={self.read_event}, write_event={self.write_event})")
    


# init
print("================INIT====================")
event_r = [
            
            Event(event_type="read", period=5, offset=0, jitter=0),
            Event(event_type="read", period=3, offset=0, jitter=0),
            Event(event_type="read", period=4, offset=0, jitter=0),
            ]

event_w = [
            
            Event(event_type="write", period=5, offset=4, jitter=0),
            Event(event_type="write", period=3, offset=3, jitter=0),
            Event(event_type="write", period=4, offset=4, jitter=0),   
            ]

for i, (r, w) in enumerate(zip(event_r, event_w)):
    r.id = i
    w.id = i

n = len(event_r)
tasks = []
for i in range(n):
    task = Task(read_event=event_r[i], write_event=event_w[i], id=i)
    tasks.append(task)

# effective_event(w1, r2)
# combine(task1, task2)
chain_asc(tasks)
# chain_desc(tasks)
chain_max_period(tasks)
chain_min_period(tasks)