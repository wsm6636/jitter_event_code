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
import random
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
    

# random events
class RandomEvent:
    def __init__(self, num_tasks, min_period, max_period, min_offset, max_offset, min_jitter, max_jitter):
        self.num_tasks = num_tasks
        self.min_period = min_period
        self.max_period = max_period
        self.min_offset = min_offset
        self.max_offset = max_offset
        self.min_jitter = min_jitter
        self.max_jitter = max_jitter
        self.tasks = self.generate_events_tasks()

    def generate_events_tasks(self):
        read_events = []
        write_events = []
        events = []
        tasks = []
        for i in range(self.num_tasks):
            # 随机生成周期
            period = random.randint(self.min_period, self.max_period)
            
            # 随机生成偏移量，确保偏移量小于周期
            read_offset = random.randint(self.min_offset, min(self.max_offset, period - 1))
            write_offset = random.randint(self.min_offset, min(self.max_offset, period - 1))
            # 随机生成抖动
            jitter = random.randint(self.min_jitter, self.max_jitter)
            
            # 创建读事件和写事件
            read_event = Event(event_type="read", period=period, offset=read_offset, jitter=jitter, id=i)
            write_event = Event(event_type="write", period=period, offset=write_offset, jitter=jitter, id=i)
            read_events.append(read_event)
            write_events.append(write_event)
            events.append((read_event, write_event))

            task = Task(read_event=read_event, write_event=write_event, id=i)
            tasks.append(task)

        return tasks
    def get_tasks(self):
        return self.tasks


# init
print("================INIT====================")



# event_r = [
            
#             Event(event_type="read", period=5, offset=0, jitter=0),
#             Event(event_type="read", period=3, offset=0, jitter=0),
#             Event(event_type="read", period=4, offset=0, jitter=0),
#             ]

# event_w = [
            
#             Event(event_type="write", period=5, offset=4, jitter=0),
#             Event(event_type="write", period=3, offset=3, jitter=0),
#             Event(event_type="write", period=4, offset=4, jitter=0),   
#             ]

# for i, (r, w) in enumerate(zip(event_r, event_w)):
#     r.id = i
#     w.id = i

# # original chain
# n = len(event_r)
# tasks = []
# for i in range(n):
#     task = Task(read_event=event_r[i], write_event=event_w[i], id=i)
#     tasks.append(task)

tasks = RandomEvent(num_tasks=3, min_period=3, max_period=5, min_offset=0, max_offset=4, min_jitter=0, max_jitter=0).get_tasks()
# # 打印任务信息
# for task in tasks:
#    print(task)