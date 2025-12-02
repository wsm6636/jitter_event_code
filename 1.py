#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-jitter chain combination for two LET tasks,
based on the algebraic ring method from:

Enrico Bini, Paolo Pazzaglia, Martina Maggio
"Zero-Jitter Chains of Periodic LET Tasks via Algebraic Rings"
IEEE Transactions on Computers

This module provides a function to combine two periodic LET tasks
into an equivalent chain task without inserting bridge tasks.
"""

def euclide_extend(a, b):
    """Extended Euclidean algorithm returning (gcd, x, y) such that a*x + b*y = gcd."""
    r0, r1 = int(a), int(b)
    s0, s1 = 1, 0
    t0, t1 = 0, 1
    while r1 != 0:
        q = r0 // r1
        r0, r1 = r1, r0 - q * r1
        s0, s1 = s1, s0 - q * s1
        t0, t1 = t1, t0 - q * t1
    return r0, s0, t0


def combine_zero_jitter(task1, task2):
    """
    Combine two LET tasks into a single equivalent chain task using zero-jitter algebraic rings.

    Args:
        task1: object with attributes .period, .read_offset, .write_offset
        task2: object with attributes .period, .read_offset, .write_offset

    Returns:
        tuple (read_event, write_event) where each event is a dict with:
            - 'offset': phase of the first occurrence
            - 'period': period of the event stream
            - 'delta': separation between consecutive events (for non-constant cases, worst-case is used)
        Returns False if combination fails (should rarely happen for valid LET tasks).
    """
    try:
        T1 = task1.period
        rd_ph1 = task1.read_offset
        wr_ph1 = task1.write_offset

        T2 = task2.period
        rd_ph2 = task2.read_offset
        wr_ph2 = task2.write_offset

        # Phase from write of task1 to read of task2
        PPhase = rd_ph2 - wr_ph1

        G, c1, c2 = euclide_extend(T1, T2)
        p1 = T1 // G
        p2 = T2 // G

        min_latency = (wr_ph1 - rd_ph1) + (wr_ph2 - rd_ph2) + (PPhase % G)

        if T1 == T2:
            T12 = T1
            rd_ph12 = rd_ph1
            wr_ph12 = wr_ph2 - PPhase + (PPhase % T1)
            rd_delta12 = T1
            wr_delta12 = T1
        elif T1 > T2:
            T12 = T1
            phi1 = (PPhase % T2) // G
            rd_ph12 = rd_ph1
            rd_delta12 = T1
            dancing = [(phi1 - j1 * p1) % p2 for j1 in range(p2)]
            wr_ph_list = [wr_ph2 - PPhase + rem * G + (PPhase % G) for rem in dancing]
            wr_delta_list = [
                (p1 // p2) * T2 if (rem >= p1 % p2) else (p1 // p2 + 1) * T2
                for rem in dancing
            ]
            inv_p1 = c1 % p2
            id_max_latency = ((phi1 + 1) * inv_p1) % p2
            wr_ph12 = wr_ph_list[id_max_latency]
            wr_delta12 = wr_delta_list[id_max_latency]
        else:  # T1 < T2
            T12 = T2
            phi2 = (PPhase % T1) // G
            wr_ph12 = wr_ph2
            wr_delta12 = T2
            dancing = [(phi2 + j2 * p2) % p1 for j2 in range(p1)]
            rd_ph_list = [rd_ph1 + PPhase - (PPhase % G) - rem * G for rem in dancing]
            rd_delta_list = [
                (p2 // p1 + 1) * T1 if (rem >= (-p2) % p1) else (p2 // p1) * T1
                for rem in dancing
            ]
            inv_p2 = c2 % p1
            id_max_latency = (-(phi2 + 1) * inv_p2) % p1
            rd_ph12 = rd_ph_list[id_max_latency]
            rd_delta12 = rd_delta_list[id_max_latency]

        read_event = {
            'offset': rd_ph12,
            'period': T12,
            'delta': rd_delta12
        }
        write_event = {
            'offset': wr_ph12,
            'period': T12,
            'delta': wr_delta12
        }

        return read_event, write_event

    except Exception:
        # In case of any numerical or logic error, fail gracefully
        return False