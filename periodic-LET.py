#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 14:26:15 2022

It implements the methods described in the paper
   Enrico Bini, Paolo Pazzaglia, Martina Maggio
   "Zero-Jitter Chains of Periodic LET Tasks via Algebraic Rings"
   submitted to IEEE Transactions on Computers

@author: Enrico Bini and friends
"""
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


# Example of Section 5.1.1
# T1 =  16
# rd_ph1 = 1
# wr_ph1 = rd_ph1+T1            # implicit deadline
# T2 = 10
# rd_ph2 = 0
# wr_ph2 = rd_ph2+T2            # implicit deadline

# Example of Section 5.2.1
# T1 = 24
# rd_ph1 = 0
# wr_ph1 = rd_ph1+T1            # implicit deadline
# T2 = 33
# rd_ph2 = 8
# wr_ph2 = rd_ph2+T2            # implicit deadline

T1 =  10
rd_ph1 = 1
wr_ph1 = rd_ph1+T1            # implicit deadline
T2 = 10
rd_ph2 = 0
wr_ph2 = rd_ph2+T2            # implicit deadline

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
   T12 = T1
   rd_ph12 = rd_ph1
   rd_delta12 = wr_delta12 = T1
   cycle = 1
   wr_ph12 = wr_ph2-PPhase+(PPhase % T1)
   max_latency = min_latency
   id_min_latency = 0
   id_max_latency = 0
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

# Now printing
print("LET Task 1")
print("\tPeriod:\t\t",T1,sep='')
print("\tRead phasing:\t",rd_ph1,sep='')
print("\tWrite phasing:\t",wr_ph1,sep='')
print("LET Task 2")
print("\tPeriod:\t\t",T2,sep='')
print("\tRead phasing:\t",rd_ph2,sep='')
print("\tWrite phasing:\t",wr_ph2,sep='')
print("Chain 1->2")
print("\tPeriod:\t\t",T12,sep='')
print("\tRead phasing:\t",rd_ph12,sep='')
print("\tWrite phasing:\t",wr_ph12,sep='')
print("\tMin latency:\t",min_latency,sep='')
print("\tMax latency:\t",max_latency,sep='')
print("\tId min latency:\t",id_min_latency,"+k*",cycle,sep='')
print("\tId max latency:\t",id_max_latency,"+k*",cycle,sep='')
print("\tRead delta:\t",rd_delta12,sep='')
print("\tWrite delta:\t",wr_delta12,sep='')
if T1 == T2:
   print("Chain 1->2 is LET",sep='')
else:
   print("Chain 1->2 becomes LET",sep='')
   if T1 > T2:
      print("\tby adding a copier LET task after Task 2")
      print("\t\tRead phasing:\t",rd_ph2next,sep='')
      print("\t\tWrite phasing:\t",rd_ph2next,sep='')
      print("\tRead phasing:\t",rd_ph12,sep='')
      print("\tWrite phasing:\t",wr_ph12[id_max_latency],sep='')
   elif T1 < T2:
      print("\tby adding a copier LET task before Task 1")
      print("\t\tRead phasing:\t",wr_ph1prev,sep='')
      print("\t\tWrite phasing:\t",wr_ph1prev,sep='')
      print("\tRead phasing:\t",rd_ph12[id_max_latency],sep='')
      print("\tWrite phasing:\t",wr_ph12,sep='')


   
   