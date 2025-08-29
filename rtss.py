import math
import utilities
import utilities.analyzer
import utilities.analysis_becker
import utilities.analyzer_our
from utilities.task import Task
from utilities.chain import CauseEffectChain
import utilities.event_simulator as es

from utilities.e2e import our_mrt_mRda_lst

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

    except Exception as e:
        print(e)
        schedule = None

    return schedule


####################################################################

t1 = Task(task_id="1",
          task_phase=2,
          task_bcet=1,
          task_wcet=3,
          task_period=5,
          task_deadline=5,
          priority=2)
t2 = Task(task_id="2",
          task_phase=4,
          task_bcet=1,
          task_wcet=1,
          task_period=5,
          task_deadline=5,
          priority=1)


# ===
task_set = [t2,t1]  # ordered by priority 
ce = [t1,t2]  # ordered in sequence
# ===

chain = CauseEffectChain(1, task_set)

ana = utilities.analyzer.Analyzer()
hyper = ana.determine_hyper_period(chain.chain)

ana.davare_single(chain)
ana.kloda(chain, hyper)
ana.reaction_duerr_single(chain)
ana.age_duerr_single(chain)
mrda_becker = utilities.analysis_becker.mrda(chain)

schedule = schedule_task_set([chain], task_set, print_status=True)
print(schedule)

# ===
# Following change_taskset_bcet:
new_task_set = [task.copy() for task in task_set]
for task in new_task_set:
    task.wcet = task.bcet

schedule_bcet = schedule_task_set([chain], new_task_set, print_status=True)
print(schedule_bcet)

mrt = utilities.analyzer_our.max_reac_local(chain, task_set, schedule, new_task_set, schedule_bcet)
let = utilities.analyzer_our.mrt_let(chain, task_set)

print(f"max reactiom time: {mrt}")
print(f"LET: {let}")
