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
      def __init__(self, name, event_type, period, offset, jitter):
         self.name = name
         self.event_type = event_type  # "read" or "write"
         self.period = period
         self.offset = offset
         self.jitter = jitter
         print(f"event series: {self.name}, {self.event_type} with period: {self.period}, offset: {self.offset}, jitter: {self.jitter}.")

class Task:
   def __init__(self, name, read_event, write_event):
      self.name = name
      self.read_event = read_event
      self.write_event = write_event
      self.period = read_event.period
      self.offset = read_event.offset
      self.jitter = 0
      print(f"task: {name}, period: {self.period}, offset: {self.offset}, read_event: {self.read_event.name}, write_event: {self.write_event.name}.")

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
   else:
      print(f"Does not exist effective write/read event series.")

   w_star = Event(name="w_star", event_type="write", period=T_star, offset=w_offser_star, jitter=w_jitter_star)
   r_star = Event(name="r_star", event_type="read", period=T_star, offset=r_offset_star, jitter=r_jitter_star)
   print(f"w_star: period: {w_star.period}, offset: {w_star.offset}, jitter: {w_star.jitter}")
   print(f"r_star: period: {r_star.period}, offset: {r_star.offset}, jitter: {r_star.jitter}")

   return (w_star, r_star)

# Algorithm 2


# init
r1 = Event(name="r1", event_type="read", period=8, offset=0, jitter=1)
w1 = Event(name="w1", event_type="write", period=8, offset=8, jitter=2)

r2 = Event(name="r2", event_type="read", period=5, offset=6, jitter=1)
w2 = Event(name="w2", event_type="write", period=5, offset=13, jitter=2)

task1 = Task(name="task1", read_event=r1, write_event=w1)
task2 = Task(name="task2", read_event=r2, write_event=w2)

effective_event(w1, r2)

   