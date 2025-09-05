"""
Borrowed some code from the following repository https://github.com/tu-dortmund-ls12-rt/end-to-end_inter

[20] M. Günzel, K.-H. Chen, N. Ueter, G. von der Brüggen, M. Dürr, and J.-J. Chen, “Timing analysis of asynchronized distributed cause- effect chains,” in Real Time and Embedded Technology and Applications Symposium (RTAS), 2021.
"""
import math
from operator import attrgetter
import utilities
import utilities.analyzer
import utilities.analyzer_our
from utilities.task import Task
from utilities.chain import CauseEffectChain
import utilities.event_simulator as es
import random
import time


"""
Copied from Paper [20]: main.py
"""
def TDA(task_set):
    """TDA analysis for a task set.
    Return True if succesful and False if not succesful."""

    ana = utilities.analyzer.Analyzer()
    # TDA.
    i = 1
    for task in task_set:
        # Prevent WCET = 0 since the scheduler can
        # not handle this yet. This case can occur due to
        # rounding with the transformer.
        if task.wcet == 0:
            raise ValueError("WCET == 0")
        task.rt = ana.tda(task, task_set[: (i - 1)])
        if task.rt > task.deadline:
            task.rt = task.deadline
        i += 1

    return True



"""
Copied from Paper [20]: main.py
"""
def schedule_task_set(ce_chains, task_set, print_status=False):

    try:
        # Preliminary: compute latency_upper_bound
        latency_upper_bound = max([ce.davare for ce in ce_chains])
        # Main part: Simulation part
        simulator = es.eventSimulator(task_set)
        # Determination of the variables used to compute the stop
        # condition of the simulation
        max_phase = max(task_set, key=lambda task: task.phase).phase
        max_period = max(task_set, key=lambda task: task.period).period
        hyper_period = utilities.analyzer.Analyzer.determine_hyper_period(task_set)

        sched_interval = (
            2 * hyper_period
            + max_phase  # interval from paper
            + latency_upper_bound  # upper bound job chain length
            + max_period
        )  # for convenience

        if print_status:
            # Information for end user.
            print("\tNumber of tasks: ", len(task_set))
            print("\tHyperperiod: ", hyper_period)
            number_of_jobs = 0
            for task in task_set:
                number_of_jobs += sched_interval / task.period
            print("\tNumber of jobs to schedule: ", "%.2f" % number_of_jobs)

        # Stop condition: Number of jobs of lowest priority task.
        simulator.dispatcher(int(math.ceil(sched_interval / task_set[-1].period)))
        
        # Simulation without early completion.
        schedule = simulator.e2e_result()
        if schedule is None:
            return False
        
    except Exception as e:
        print(e)
        schedule = None

    return schedule


"""
Random utilization generation by UUniFast, Bini et al 2005
"""
def uunifast(n, u):
    utilizations = []
    sumU = u
    for i in range(1, n):
        nextSumU = sumU * random.random() ** (1.0 / (n - i))
        utilizations.append(sumU - nextSumU)
        sumU = nextSumU
    utilizations.append(sumU)
    return utilizations


"""
NEWFUNC by shumo. 
"""
def run_analysis_Gunzel_LET(num_tasks, periods,read_offsets,write_offsets, per_jitter):
    """
    Convert the task set of our paper to the LET model of paper [20] and calculate its LET value.
    """
    task_set = []
    for i, period, read_offset, write_offset in zip(range(num_tasks), periods, read_offsets, write_offsets):
        maxjitter = per_jitter * period          
        bcet      = write_offset - read_offset
        wcet      = maxjitter + bcet
        deadline  = period  

        t = Task(task_id=i,
                task_phase=read_offset,
                task_bcet=bcet,
                task_wcet=wcet,
                task_period=period,
                task_deadline=deadline,
                priority=i)

        task_set.append(t)

    # RM
    task_set = sorted(task_set, key=attrgetter('period'))
    for i, t in enumerate(task_set):
        t.priority = i  
        
    chain = CauseEffectChain(1, task_set)
    ana = utilities.analyzer.Analyzer()
    ana.davare_single(chain)

    let = utilities.analyzer_our.mrt_let(chain, task_set)
    
    return  let


"""
NEWFUNC by shumo. 
"""
def run_analysis_Gunzel_IC(num_tasks, periods):
    """
    Schedule results of paper [20]
    For each generated task set (Kramer's periods [40], utilizations with UUniFast with total utilization of 50%, single CPU)
    """
    task_set = []
    totU = 0.5
    vecU = uunifast(num_tasks, totU)
    selected_periods = random.choices(periods,  k=num_tasks)
    for i, period in zip(
        range(num_tasks), selected_periods):
        wcet      = vecU[i]*period    
        bcet      = 1*wcet   
        deadline  = period  

        t = Task(task_id=i,
                task_phase= random.randint(0, period//2),  # random phase
                task_bcet=bcet,
                task_wcet=wcet,
                task_period=period,
                task_deadline=deadline)
        task_set.append(t)

    # Since the "eventSimulator" in "utilities/event_simulator.py" requires the priority order to be the same as the task order
    # we use RM
    task_set = sorted(task_set, key=attrgetter('period'))
    for i, t in enumerate(task_set):
        t.priority = i  
    
    # Copied from Paper [20],
    # Prerequisites for calculating response times and bounds
    chain = CauseEffectChain(1, task_set)
    ana = utilities.analyzer.Analyzer()
    ana.davare_single(chain)
    TDA(task_set)

    # Get the schedule_wcet
    # then change to taskset_bcet for the schedule_bcet
    schedule_wcet = schedule_task_set([chain], task_set, print_status=False)
    new_task_set = [task.copy() for task in task_set]
    for task in new_task_set:
        task.wcet = task.bcet
    schedule_bcet = schedule_task_set([chain], new_task_set, print_status=False)
    
    t_0 = time.perf_counter()
    ic = utilities.analyzer_our.max_reac_local(chain, task_set, schedule_wcet, new_task_set, schedule_bcet)
    t_1 = time.perf_counter()
    runtime = t_1 - t_0

    sorted_periods = [t.period for t in task_set]

    return ic, sorted_periods, schedule_wcet, task_set, schedule_bcet, new_task_set,runtime

if __name__ == "__main__":
    run_analysis_Gunzel_IC(2, [5])
    run_analysis_Gunzel_LET(4,[2,1,5,3],[0,0,0,0],[2,1,5,3],0)