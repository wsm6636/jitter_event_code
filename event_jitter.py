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
import datetime
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
        self.read_time = 0
        self.write_time = 0
        # print(f"event {self.event_type}_{self.id},  period: {self.period}, offset: {self.offset}, jitter: {self.jitter}.")
    def __repr__(self):
        return (f"Event(type={self.event_type},id={self.id}, period={self.period}, "
            f"offset={self.offset}, jitter={self.jitter}, read_time={self.read_time:.2f}, "
            f"write_time={self.write_time:.2f}")
    def get_trigger_time(self, j):
        random_jitter = random.uniform(0, self.jitter)
        tj = j * self.period + self.offset + random_jitter
        # print(f"event {self.event_type}_{self.id}, j: {j}, trigger_time: {tj:.2f}.")
        return tj

class Task:
    def __init__(self, read_event, write_event, id=None):
        self.id = id
        self.read_event = read_event
        self.write_event = write_event
        self.period = read_event.period
        self.offset = read_event.offset
        self.jitter = 0
        # print(f"task_{self.id}, period: {self.period}, offset: {self.offset}, read_event: {self.read_event.event_type}_{self.read_event.id}, write_event: {self.write_event.event_type}_{self.write_event.id}.")
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
            # LET
            write_offset = period
            # write_offset = random.randint(self.min_offset, min(self.max_offset, period - 1))
            
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
    # def get_tasks(self):
    #     return self.tasks

# Euclide's algorithm for coefficients of Bezout's identity
def euclide_extend(a,b):
    r0 = int(a)
    r1 = int(b)
    s0 = 1
    s1 = 0
    t0 = 0
    t1 = 1
    while (r1 != 0):
        q = r0 // r1
        new_r = r0 % r1
        new_s = s0-q*s1
        new_t = t0-q*t1
        r0 = r1
        s0 = s1
        t0 = t1
        r1 = new_r
        s1 = new_s
        t1 = new_t
    return (r0,s0,t0)

# effective event
# Algorithm 2 line 1
def effective_event(w, r):
   w_star = None
   r_star = None
   delta = r.offset - w.offset
#    print(f"delta: {delta}.")
   (G,pw,pr) = euclide_extend(w.period, r.period)
   # print(f"G: {G}.")
   T_star = max(w.period, r.period)
#    print(f"T_star: {T_star}.")

   if w.period == r.period: # Theorem 12
    #   print(f"periods are equal. Theorem 12.")
      if w.jitter <= (delta % T_star) & (delta % T_star) < (T_star - r.jitter): # Formula (14)
         w_jitter_star = w.jitter
         r_jitter_star = r.jitter  # Formula (15)
         if delta < 0:
            # print(f"delta < 0. Formula (15).")  
            w_offser_star = w.offset
            r_offset_star = w.offset + (delta % T_star) # Formula (15)
         else:
            # print(f"delta >= 0. Formula (15).")
            w_offser_star = r.offset - (delta % T_star) # Formula (15)
            r_offset_star = r.offset
      else:
         print(f"Does not conform to Theorem 12, Formula (14).")
         return False
   elif w.period > r.period:
      if w.jitter == r.jitter == 0: # Lemma (15)
        #  print(f"w.period > r.period, without jitter. Lemma (15), Formula (28).")
         kw = max (0, ((delta - r.period) // w.period) + 1)
         w_offser_star = w.offset + kw*w.period
         w_jitter_star = 0                            
         r_offset_star = w_offser_star + (delta % G)
         r_jitter_star = r.period - G               # Formula (28)
      elif (r.period + r.jitter) <= (w.period - w.jitter): # Formula (17) Theorem (13)
        #  print(f"w.period > r.period, with jitter. Theorem (13), Formula (17).")
         kw = max (0, ((delta + r.jitter - r.period) // w.period) + 1) # Formula (19)
         w_offser_star = w.offset + kw*w.period
         w_jitter_star = w.jitter
         r_offset_star = w_offser_star
         r_jitter_star = r.period + w.jitter # Formula (18) 
      else:
         print(f"Does not conform to Theorem (13), Formula (17).")
         return False
   elif w.period < r.period:
      if w.jitter == r.jitter == 0: # Lemma (16)
        #  print(f"w.period < r.period, without jitter. Lemma (16), Formula (30).")
         kr = max (0, math.ceil(-delta / r.period))
         r_offset_star = r.offset + kr*r.period
         r_jitter_star = 0
         w_offser_star = r_offset_star - (delta % G) - w.period + G
         w_jitter_star = w.period - G                                 # Formula (30)
      elif (w.period + w.jitter) <= (r.period - r.jitter): # Formula (22) Theorem (14)
        #  print(f"w.period < r.period, with jitter. Theorem (14), Formula (22).")
         kr = max (0, math.ceil((w.jitter - delta) / r.period)) # Formula (25) 
         r_offset_star = r.offset + kr*r.period
         r_jitter_star = r.jitter                     # Formula (23)
         w_offser_star = r_offset_star - w.period
         w_jitter_star = w.period + r.jitter          # Formula (24)
      else:
         print(f"Does not conform to Theorem (14), Formula (22).")
         return False
   else:
      print(f"Does not exist effective write/read event series.")
      return False

   w_star = Event(id=w.id, event_type="write_star", period=T_star, offset=w_offser_star, jitter=w_jitter_star)
   r_star = Event(id=r.id, event_type="read_star", period=T_star, offset=r_offset_star, jitter=r_jitter_star)
   # print(f"w_star : period: {w_star.period}, offset: {w_star.offset}, jitter: {w_star.jitter}")
   # print(f"r_star : period: {r_star.period}, offset: {r_star.offset}, jitter: {r_star.jitter}")

   return (w_star, r_star)

# Algorithm 2
def combine(task1,task2):
   r1=task1.read_event
   w1=task1.write_event
   r2=task2.read_event
   w2=task2.write_event
#    print("================EFFECTIVE====================")
#    print(f"effective_event of  ({w1.event_type}{w1.id},{r2.event_type}{r2.id})")

   result = effective_event(w1, r2)
   if result:
      (w1_star, r2_star) = result
   elif w1.jitter != 0 or r2.jitter != 0:
    #   print("==============TRY JITTER FREE==============")
      if w1.jitter != 0 and r2.jitter == 0:
         w1_free = jitter_free(w1)
         result = effective_event(w1_free, r2)
         if result:
            # print("==============w1_free==============")
            (w1_star, r2_star) = result
            w1 = w1_free
      elif w1.jitter==0 and r2.jitter != 0:
         r2_free = jitter_free(r2)
         result = effective_event(w1, r2_free)
         if result:
            # print("==============r2_free==============")
            (w1_star, r2_star) = result     
            r2 = r2_free
      elif w1.jitter != 0 and r2.jitter != 0:
         w1_free = jitter_free(w1)
         r2_free = jitter_free(r2)
         result = effective_event(w1_free, r2_free)
         if result:
            # print("==============w1_free, r2_free==============")
            (w1_star, r2_star) = result
            w1 = w1_free
            r2 = r2_free
   else:
      print("==========FAILED TO EFFECTIVE EVENT==========")
      return False

   # print(f"w_star : period: {w1_star.period}, offset: {w1_star.offset}, jitter: {w1_star.jitter}")
   # print(f"r_star : period: {r2_star.period}, offset: {r2_star.offset}, jitter: {r2_star.jitter}")

   # print(f"r1: period: {r1.period}, offset: {r1.offset}, jitter: {r1.jitter}")
   # print(f"w1: period: {w1.period}, offset: {w1.offset}, jitter: {w1.jitter}")
   # print(f"r2: period: {r2.period}, offset: {r2.offset}, jitter: {r2.jitter}")
   # print(f"w2: period: {w2.period}, offset: {w2.offset}, jitter: {w2.jitter}")

   T_star = w1_star.period   #line 2
   if task1.period > task2.period: # line 4
      r_1_2_offset = r1.offset + w1_star.offset - w1.offset #line 5
      r_1_2_jitter = r1.jitter  #line 6
      m2 = w2.offset - r2.offset - r2.jitter
      M2 = w2.offset - r2.offset + w2.jitter  #line 7
      w_1_2_offset = r2_star.offset + m2 
      w_1_2_jitter = r2_star.jitter + M2 - m2 #line 8
   elif task1.period < task2.period: #line 9
      w_1_2_offset = w2.offset + r2_star.offset - r2.offset #line 10
      w_1_2_jitter = w2.jitter  #line 11
      m1 = w1.offset - r1.offset - r1.jitter
      M1 = w1.offset - r1.offset + w1.jitter #line 12
      r_1_2_offset = w1_star.offset - M1
      r_1_2_jitter = w1_star.jitter + M1 - m1  #line 13   
   else: #line 14
      r_1_2_offset = r1.offset + w1_star.offset - w1.offset
      r_1_2_jitter = r1.jitter
      w_1_2_offset = w2.offset + r2_star.offset - r2.offset
      w_1_2_jitter = w2.jitter

   combined_id = f"{task1.id}_{task2.id}"
   r_1_2 = Event(id=combined_id ,event_type="read_combined", period=T_star, offset=r_1_2_offset, jitter=r_1_2_jitter) #line 19
   w_1_2 = Event(id=combined_id ,event_type="write_combined", period=T_star, offset=w_1_2_offset, jitter=w_1_2_jitter) #line 20
   # print(f"period: {r_1_2.period}, offset: {r_1_2.offset}, jitter: {r_1_2.jitter}")
   # print(f"period: {w_1_2.period}, offset: {w_1_2.offset}, jitter: {w_1_2.jitter}")
   return (r_1_2, w_1_2)

def combine_no_free_jitter(task1,task2):
   r1=task1.read_event
   w1=task1.write_event
   r2=task2.read_event
   w2=task2.write_event

   result = effective_event(w1, r2)
   # print(result)
   if result:
      (w1_star, r2_star) = result
   else:
      print("==========FAILED TO EFFECTIVE EVENT==========")
      return False
   T_star = w1_star.period   #line 2
   if task1.period > task2.period: # line 4
      r_1_2_offset = r1.offset + w1_star.offset - w1.offset #line 5
      r_1_2_jitter = r1.jitter  #line 6
      m2 = w2.offset - r2.offset - r2.jitter
      M2 = w2.offset - r2.offset + w2.jitter  #line 7
      w_1_2_offset = r2_star.offset + m2 
      w_1_2_jitter = r2_star.jitter + M2 - m2 #line 8
   elif task1.period < task2.period: #line 9
      w_1_2_offset = w2.offset + r2_star.offset - r2.offset #line 10
      w_1_2_jitter = w2.jitter  #line 11
      m1 = w1.offset - r1.offset - r1.jitter
      M1 = w1.offset - r1.offset + w1.jitter #line 12
      r_1_2_offset = w1_star.offset - M1
      r_1_2_jitter = w1_star.jitter + M1 - m1  #line 13   
   else: #line 14
      r_1_2_offset = r1.offset + w1_star.offset - w1.offset
      r_1_2_jitter = r1.jitter
      w_1_2_offset = w2.offset + r2_star.offset - r2.offset
      w_1_2_jitter = w2.jitter

   combined_id = f"{task1.id}_{task2.id}"
   r_1_2 = Event(id=combined_id ,event_type="read_combined", period=T_star, offset=r_1_2_offset, jitter=r_1_2_jitter) #line 19
   w_1_2 = Event(id=combined_id ,event_type="write_combined", period=T_star, offset=w_1_2_offset, jitter=w_1_2_jitter) #line 20
   # print(f"period: {r_1_2.period}, offset: {r_1_2.offset}, jitter: {r_1_2.jitter}")
   # print(f"period: {w_1_2.period}, offset: {w_1_2.offset}, jitter: {w_1_2.jitter}")
   return (r_1_2, w_1_2)


#jitter-free Figure 2
def jitter_free(event):
      if event.jitter == 0:
         return event
      #   print(f"{event.event_type}{event.id} is zero jitter.")
      else:
         event.jitter = 0
         event.offset = event.offset + event.jitter
      return event

#e2e
def e2e(r,w):
#    print("================E2E====================")
   min_e2e = w.offset - r.offset - r.jitter
   max_e2e = w.offset + w.jitter - r.offset
#    print(f"min_e2e: {min_e2e}, max_e2e: {max_e2e}")
   return (min_e2e, max_e2e)

#chain: sort by asc index
def chain_asc(tasks):
#    print("================CHAIN_ASC====================")
   n = len(tasks)
   current_task = tasks[0]

   for i in range(1, n):  
    #   print(f"================Combining task {current_task.id} and {tasks[i].id}====================")
      result = combine(current_task, tasks[i])
      if result is False:
        #  print("================CHAIN_ASC END====================")
         print(f"Failed to combine task {current_task.id} and task {tasks[i].id}.")
         return False
      else:
         (r,w) = result
        #  print("================UPDATE combined task====================")
         current_task = Task(read_event=r, write_event=w, id=r.id)

   return e2e(r,w), r, w, current_task


#chain: sort by desc index
def chain_desc(tasks):
#    print("================CHAIN_DESC====================")
   n = len(tasks)
   current_task = tasks[-1]

   for i in range(n-2, -1,-1):  
    #   print(f"================Combining task {tasks[i].id} and {current_task.id}====================")
      result = combine(tasks[i], current_task)
      if result is False:
        #  print("================CHAIN_DESC END====================")
         print(f"Failed to combine task {tasks[i].id} and task {current_task.id}.")
         return False
      else:
         (r,w) = result
        #  print("================UPDATE combined task====================")
         current_task = Task(read_event=r, write_event=w, id=r.id)

   return e2e(r,w),  r, w, current_task

def chain_asc_no_free_jitter(tasks):
#    print("================CHAIN_ASC====================")
   n = len(tasks)
   current_task = tasks[0]

   for i in range(1, n):  
    #   print(f"================Combining task {current_task.id} and {tasks[i].id}====================")
      result = combine_no_free_jitter(current_task, tasks[i])
      if result is False:
        #  print("================CHAIN_ASC END====================")
         print(f"Failed to combine task {current_task.id} and task {tasks[i].id}.")
         return False
      else:
         (r,w) = result
        #  print("================UPDATE combined task====================")
         current_task = Task(read_event=r, write_event=w, id=r.id)

   return e2e(r,w), r, w, current_task

#chain max period order
def chain_max_period(tasks):
#    print("================CHAIN_MAX_PERIOD====================")
   max_period_task = max(tasks, key=lambda x: x.period) #taski
   max_period_index = tasks.index(max_period_task)
#    print(f"max_period = {max_period_task.period}, task_id = {max_period_task.id}")

   #Grouping
   predecessor_group = tasks[:max_period_index + 1]  #task0~i
   successor_group = tasks[max_period_index:]       #taski~n
   final_tasks = []
   final_e2es = []
   #task0~i chain
#    print("===============Processing Predecessor Group (desc)===============")
   if len(predecessor_group) > 1:
      predecessor_result = chain_desc(predecessor_group)
      predecessor_task_e2e, r_predecessor, w_predecessor, predecessor_task = predecessor_result
      final_tasks.append(predecessor_task)
      final_e2es.append(predecessor_task_e2e)
   else:
      print(f"predecessor chain is only one task.")

   #taski~n chain
#    print("===============Processing Successor Group (asc)===============")
   if len(successor_group) > 1:
      successor_result = chain_asc(successor_group)
      successor_task_e2e, r_successor, w_successor, successor_task = successor_result
      final_tasks.append(successor_task)
      final_e2es.append(successor_task_e2e)
   else:
      print(f"successor chain is only one task.")

   # print("===============Combining Predecessor and Successor Results===============")
   if len(final_tasks) == 1:
      print(f"================Final: Only one task in the final chain================")
      print(f"final_e2e: min_e2e: {final_e2es[0][0]}, max_e2e: {final_e2es[0][1]}, max_reaction_time: {final_e2es[0][1] + final_tasks[0].read_event.period}, min_reaction_time: {final_e2es[0][0] + final_tasks[0].read_event.period}")
      print(f"final_r: period: {final_tasks[0].read_event.period}, offset: {final_tasks[0].read_event.offset}, jitter: {final_tasks[0].read_event.jitter}")
      print(f"final_w: period: {final_tasks[0].write_event.period}, offset: {final_tasks[0].write_event.offset}, jitter: {final_tasks[0].write_event.jitter}")
      return True
   elif len(final_tasks) > 1:
      final_combine_result = chain_asc(final_tasks)
      if final_combine_result:
         final_e2e, final_r, final_w, final_task = final_combine_result
         print("================Final Combined Result====================")
         print(f"final_e2e: min_e2e: {final_e2e[0]}, max_e2e: {final_e2e[1]}, max_reaction_time: {final_e2e[1] + final_r.period}, min_reaction_time: {final_e2e[0] + final_r.period}")
         print(f"final_r: period: {final_r.period}, offset: {final_r.offset}, jitter: {final_r.jitter}")
         print(f"final_w: period: {final_w.period}, offset: {final_w.offset}, jitter: {final_w.jitter}")
         return True
   else:
    #   print("Failed to combine predecessor and successor results.")
      return False

#chain min period order
def chain_min_period(tasks):
#    print("================CHAIN_MIN_PERIOD====================")
   min_period_task = min(tasks, key=lambda x: x.period) #taski
   min_period_index = tasks.index(min_period_task)
#    print(f"min_period = {min_period_task.period}, task_id = {min_period_task.id}")

   #Grouping
   predecessor_group = tasks[:min_period_index + 1]  #task0~i
   successor_group = tasks[min_period_index:]       #taski~n
   final_tasks = []
   final_e2es = []
   #task0~i chain
#    print("===============Processing Predecessor Group (desc)===============")
   if len(predecessor_group) > 1:
      predecessor_result = chain_desc(predecessor_group)
      predecessor_task_e2e, r_predecessor, w_predecessor, predecessor_task = predecessor_result
      final_tasks.append(predecessor_task)
      final_e2es.append(predecessor_task_e2e)
   else:
      print(f"predecessor chain is only one task.")

   #taski~n chain
#    print("===============Processing Successor Group (asc)===============")
   if len(successor_group) > 1:
      successor_result = chain_asc(successor_group)
      successor_task_e2e, r_successor, w_successor, successor_task = successor_result
      final_tasks.append(successor_task)
      final_e2es.append(successor_task_e2e)
   else:
      print(f"successor chain is only one task.")

   # print("===============Combining Predecessor and Successor Results===============")
   if len(final_tasks) == 1:
      print(f"================Final: Only one task in the final chain================")
      print(f"final_e2e: min_e2e: {final_e2es[0][0]}, max_e2e: {final_e2es[0][1]}, max_reaction_time: {final_e2es[0][1] + final_tasks[0].read_event.period}, min_reaction_time: {final_e2es[0][0] + final_tasks[0].read_event.period}")
      print(f"final_r: period: {final_tasks[0].read_event.period}, offset: {final_tasks[0].read_event.offset}, jitter: {final_tasks[0].read_event.jitter}")
      print(f"final_w: period: {final_tasks[0].write_event.period}, offset: {final_tasks[0].write_event.offset}, jitter: {final_tasks[0].write_event.jitter}")
      return True
   elif len(final_tasks) > 1:
      final_combine_result = chain_asc(final_tasks)
      if final_combine_result:
         final_e2e, final_r, final_w, final_task = final_combine_result
         print("================Final Combined Result====================")
         print(f"final_e2e: min_e2e: {final_e2e[0]}, max_e2e: {final_e2e[1]}, max_reaction_time: {final_e2e[1] + final_r.period}, min_reaction_time: {final_e2e[0] + final_r.period}")
         print(f"final_r: period: {final_r.period}, offset: {final_r.offset}, jitter: {final_r.jitter}")
         print(f"final_w: period: {final_w.period}, offset: {final_w.offset}, jitter: {final_w.jitter}")
         return True
   else:
      print("Failed to combine predecessor and successor results.")
      return False
   
def our_chain(tasks):
   final_combine_result = chain_asc_no_free_jitter(tasks)
   if final_combine_result:
      final_e2e, final_r, final_w, final_task = final_combine_result
      print(f"final_e2e: min_e2e: {final_e2e[0]}, max_e2e: {final_e2e[1]}, max_reaction_time: {final_e2e[1] + final_r.period}, min_reaction_time: {final_e2e[0] + final_r.period}")
      print(f"final_r: period: {final_r.period}, offset: {final_r.offset}, jitter: {final_r.jitter}")
      print(f"final_w: period: {final_w.period}, offset: {final_w.offset}, jitter: {final_w.jitter}")
      return True
   else:
   #   print("Failed to combine predecessor and successor results.")
      return False
   

# def is_valid_chain(task_chain):
#     return True


def find_valid_task_chains(tasks):
   valid_chains = []
   for start_instance in range(10):  # 从不同的起始实例编号开始
      task_chain = []
      last_write_time = -float('inf')
      for task in tasks:
         read_event = task.read_event
         write_event = task.write_event

         # 从当前起始实例编号开始
         read_instance = start_instance
         while True:
               read_time = read_event.get_trigger_time(read_instance)
               if read_time >= last_write_time:
                  break
               read_instance += 1

         # 找到第一个满足条件的写事件实例编号
         write_instance = read_instance  # 从读事件的实例编号开始
         write_time = write_event.get_trigger_time(write_instance)

         read_event.read_time = read_time
         write_event.write_time = write_time
         task_chain.append((read_event, read_time, read_instance))
         task_chain.append((write_event, write_time, write_instance))
         last_write_time = write_time

      # 检查生成的任务链是否有效
      if len(task_chain) == len(tasks) * 2:
         valid_chains.append(task_chain)

   return valid_chains


def calculate_max_reaction_time(valid_chains):
   max_reaction_time = 0
   min_reaction_time = float('inf')
   for chain in valid_chains:
      first_read_event = chain[0]
      last_write_event = chain[-1]
      reaction_time = last_write_event[1] - first_read_event[1] + first_read_event[0].period
      max_reaction_time = max(max_reaction_time, reaction_time)
      min_reaction_time = min(min_reaction_time, reaction_time)
   print(f"Max reaction time: {max_reaction_time:.2f}")
   print(f"Min reaction time: {min_reaction_time:.2f}")
   return max_reaction_time, min_reaction_time



def write_results_to_file(tasks, valid_chains):
   # 获取当前时间戳
   timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
   filename = f"task_chain_results_{timestamp}.txt"
   
   with open(filename, "w") as file:
      # 写入所有事件和任务的信息
      file.write("All Events and Tasks Information:\n")
      for task in tasks:
         file.write(f"Task {task.id}:\n")
         file.write(f"   Read Event: {task.read_event}\n")
         file.write(f"   Write Event: {task.write_event}\n")
      file.write("\n")

      # 写入任务链结果
      if not valid_chains:
         file.write("No valid task chains found.\n")
      else:
         for i, task_chain in enumerate(valid_chains):
               file.write(f"Valid Task Chain {i}:\n")
               for j, (event, time, instance) in enumerate(task_chain):
                  file.write(f"   Event {j}: {event.event_type}_{event.id}_{instance} at {time:.2f}\n")
               file.write(f"   Reaction Time: {task_chain[-1][1] - task_chain[0][1] + task_chain[0][0].period:.2f}\n")
               file.write(f"\n")
      file.write(f"Maximum Reaction Time for all task chains: {max_reaction_time:.2f}\n")
      file.write(f"Global Maximum Reaction Time: {global_max_reaction_time:.2f}\n")
   print(f"Results written to {filename}")



def calculate_reaction_time(task_chain):
   first_read_time = task_chain[1]
   first_read_period = task_chain[0]
   last_write_time = task_chain[-1]
   return last_write_time - first_read_time + first_read_period

def objective_function(x):
   return -calculate_reaction_time(x)  # 负号用于将最大化问题转换为最小化问题

def maximize_reaction_time(valid_chains):
   initial_x = []
   if not valid_chains:
      return 0
   initial_task_chain = valid_chains[0]

   for event, _, _ in initial_task_chain:
      initial_x.append(event.period,)  # 提取周期
      initial_x.append(event.read_time)  # 提取读时间
      initial_x.append(event.write_time) 
   print(f"Initial task chain: {initial_x}")
   
   result = basinhopping(objective_function, initial_x, niter=10, T=1.0, stepsize=1)
   max_reaction_time = -result.fun  # 负号用于将最小化结果转换为最大化结果
   print(f"basinhopping Maximized reaction time: {max_reaction_time:.2f}")
   return max_reaction_time



# init
print("================INIT====================")


tasks = RandomEvent(num_tasks=3, min_period=3, max_period=8, 
                                    min_offset=0, max_offset=5, min_jitter=0, max_jitter=2).tasks 

print("================AG2====================")
# effective_event(w1, r2)
# combine(task1, task2)
# chain_asc(tasks)
# chain_desc(tasks)
# chain_max_period(tasks)
# chain_min_period(tasks)
our = our_chain(tasks)
if our is False:
   print("END.")
else:
   print("================OTHER====================")
   valid_chains = find_valid_task_chains(tasks)
   max_reaction_time = calculate_max_reaction_time(valid_chains)
   global_max_reaction_time = maximize_reaction_time(valid_chains)

# 将结果写入文件
# write_results_to_file(tasks, valid_chains)