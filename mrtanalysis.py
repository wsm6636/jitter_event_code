import math
import utilities
import utilities.analyzer
import utilities.analysis_becker
import utilities.analyzer_our
from utilities.task import Task
from utilities.chain import CauseEffectChain
import utilities.event_simulator as es

# from main import our_mrt_mRda_lst


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
        # if task.rt > task.deadline:
            # task.rt = task.deadline
        i += 1

    return True

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


def G2023_analysis(num_tasks, periods,read_offsets,write_offsets, per_jitter):
    task_set = []
    ce = []
    for i, period, read_offset, write_offset in zip(
        range(num_tasks), periods, read_offsets, write_offsets):

        maxjitter = per_jitter * period          
        bcet      = write_offset - read_offset
        wcet      = maxjitter + bcet
        # deadline  = write_offset  + maxjitter 
        deadline  = period  
        # rt        = write_offset + maxjitter

        t = Task(task_id=i,
                task_phase=read_offset,
                task_bcet=bcet,
                task_wcet=wcet,
                task_period=period,
                task_deadline=deadline,
                priority=i)

        task_set.append(t)
        ce.append(t)

    # for task in task_set:
    #     print(task)

    chain = CauseEffectChain(1, task_set)

    ana = utilities.analyzer.Analyzer()
    hyper = ana.determine_hyper_period(chain.chain)

    ana.davare_single(chain)
    ana.kloda(chain, hyper)
    ana.reaction_duerr_single(chain)
    ana.age_duerr_single(chain)
    mrda_becker = utilities.analysis_becker.mrda(chain)

    schedule = schedule_task_set([chain], task_set, print_status=False)
    # print(f"wcet : {schedule}")
    if schedule is False:
        return None, None
    
    TDA(task_set)
    # ===
    # Following change_taskset_bcet:
    new_task_set = [task.copy() for task in task_set]
    for task in new_task_set:
        task.wcet = task.bcet

    schedule_bcet = schedule_task_set([chain], new_task_set, print_status=False)
    # print(f"bcet: {schedule_bcet}")

    mrt = utilities.analyzer_our.max_reac_local(chain, task_set, schedule, new_task_set, schedule_bcet)
    let = utilities.analyzer_our.mrt_let(chain, task_set)

    # print(f"max reactiom time: {mrt}")
    # print(f"LET: {let}")

    return mrt, let


if __name__ == "__main__":
    mrt, let = G2023_analysis(2, [5,5],[2,4],[3,5],0.1)