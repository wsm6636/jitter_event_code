#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 05 10:38:03 2025 

It implements the methods described in the paper
   Shumo Wang, Enrico Bini, Martina Maggio
   "Understanding Jitter Propagation in Task Chains"

@author: Shumo Wang
"""

# Event and Task classes
import math


class Event:
   def __init__(self, event_type, period, offset, jitter,id=None):
      self.id = id
      self.event_type = event_type  # "read" or "write"
      self.period = period
      self.offset = offset
      self.jitter = jitter
      print(f"event {self.event_type}{self.id},  period: {self.period}, offset: {self.offset}, jitter: {self.jitter}.")
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
      print(f"task{self.id}, period: {self.period}, offset: {self.offset}, read_event: {self.read_event.event_type}{self.read_event.id}, write_event: {self.write_event.event_type}{self.write_event.id}.")
      def __repr__(self):
         return (f"Task(period={self.period}, offset={self.offset}, "
               f"read_event={self.read_event}, write_event={self.write_event})")

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
   print(f"delta: {delta}.")
   (G,pw,pr) = euclide_extend(w.period, r.period)
   print(f"G: {G}.")
   T_star = max(w.period, r.period)
   print(f"T_star: {T_star}.")

   if w.period == r.period: # Theorem 12
      print(f"periods are equal. Theorem 12.")
      if w.jitter <= (delta % T_star) & (delta % T_star) < (T_star - r.jitter): # Formula (14)
         w_jitter_star = w.jitter
         r_jitter_star = r.jitter  # Formula (15)
         if delta < 0:
            print(f"delta < 0. Formula (15).")  
            w_offser_star = w.offset
            r_offset_star = w.offset + (delta % T_star) # Formula (15)
         else:
            print(f"delta >= 0. Formula (15).")
            w_offser_star = r.offset - (delta % T_star) # Formula (15)
            r_offset_star = r.offset
      else:
         print(f"Does not conform to Theorem 12, Formula (14).")
         return False
   elif w.period > r.period:
      if w.jitter == r.jitter == 0: # Lemma (15)
         print(f"w.period > r.period, without jitter. Lemma (15), Formula (28).")
         kw = max (0, ((delta - r.period) // w.period) + 1)
         w_offser_star = w.offset + kw*w.period
         w_jitter_star = 0                            
         r_offset_star = w_offser_star + (delta % G)
         r_jitter_star = r.period - G               # Formula (28)
      elif (r.period + r.jitter) <= (w.period - w.jitter): # Formula (17) Theorem (13)
         print(f"w.period > r.period, with jitter. Theorem (13), Formula (17).")
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
         print(f"w.period < r.period, without jitter. Lemma (16), Formula (30).")
         kr = max (0, math.ceil(-delta / r.period))
         r_offset_star = r.offset + kr*r.period
         r_jitter_star = 0
         w_offser_star = r_offset_star - (delta % G) - w.period + G
         w_jitter_star = w.period - G                                 # Formula (30)
      elif (w.period + w.jitter) <= (r.period - r.jitter): # Formula (22) Theorem (14)
         print(f"w.period < r.period, with jitter. Theorem (14), Formula (22).")
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
   print("================EFFECTIVE====================")
   print(f"effective_event of  ({w1.event_type}{w1.id},{r2.event_type}{r2.id})")

   result = effective_event(w1, r2)
   if result:
      (w1_star, r2_star) = result
   elif w1.jitter != 0 or r2.jitter != 0:
      print("==============TRY JITTER FREE==============")
      if w1.jitter != 0 and r2.jitter == 0:
         w1_free = jitter_free(w1)
         result = effective_event(w1_free, r2)
         if result:
            print("==============w1_free==============")
            (w1_star, r2_star) = result
            w1 = w1_free
      elif w1.jitter==0 and r2.jitter != 0:
         r2_free = jitter_free(r2)
         result = effective_event(w1, r2_free)
         if result:
            print("==============r2_free==============")
            (w1_star, r2_star) = result     
            r2 = r2_free
      else:
         result = effective_event(w1_free, r2_free)
         if result:
            print("==============w1_free, r2_free==============")
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

   r_1_2 = Event(id=task2.id ,event_type="read_combined", period=T_star, offset=r_1_2_offset, jitter=r_1_2_jitter) #line 19
   w_1_2 = Event(id=task2.id ,event_type="write_combined", period=T_star, offset=w_1_2_offset, jitter=w_1_2_jitter) #line 20
   # print(f"period: {r_1_2.period}, offset: {r_1_2.offset}, jitter: {r_1_2.jitter}")
   # print(f"period: {w_1_2.period}, offset: {w_1_2.offset}, jitter: {w_1_2.jitter}")
   return (r_1_2, w_1_2)

#jitter-free Figure 2
def jitter_free(event):
   if event.jitter == 0:
      print(f"{event.event_type}{event.id} is zero jitter.")
   else:
      event.jitter = 0
      event.offset = event.offset + event.jitter
   return event

#e2e
def e2e(r,w):
   print("================E2E====================")
   min_e2e = w.offset - r.offset - r.jitter
   max_e2e = w.offset + w.jitter - r.offset
   print(f"min_e2e: {min_e2e}, max_e2e: {max_e2e}")
   return (min_e2e, max_e2e)

#chain
def chain(tasks):
   print("================CHAIN====================")
   n = len(tasks)
   current_task = tasks[0]

   for i in range(1, n):  
      print(f"================Combining task {current_task.id} and {tasks[i].id}====================")
      result = combine(current_task, tasks[i])
      if result is False:
         print("================END====================")
         print(f"Failed to combine task {current_task.id} and task {tasks[i].id}.")
         return False
      else:
         (r,w) = result
         print("================UPDATE combined task====================")
         current_task = Task(read_event=r, write_event=w, id=i)

   return e2e(r,w)

# init
print("================INIT====================")
event_r = [
            Event(event_type="read", period=8, offset=0, jitter=1),
            Event(event_type="read", period=5, offset=6, jitter=1),
            Event(event_type="read", period=4, offset=0, jitter=0),
            ]

event_w = [
            Event(event_type="write", period=8, offset=8, jitter=2),
            Event(event_type="write", period=5, offset=13, jitter=2),
            Event(event_type="write", period=4, offset=3, jitter=0)
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
chain(tasks)
   