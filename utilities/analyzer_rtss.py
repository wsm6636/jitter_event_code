"""RTSS
"""
import sys
import os
import itertools
import math
import event_simulator as es
import analyzer as a


class CauseEffectChain:
    """Cause-effect chain."""

    def __init__(self, id, chain, interconnected=[]):
        """Initialize a cause-effect chain."""
        self.id = id  # unique identifier
        self.chain = chain  # list of all tasks in the chain
        # List of local cause-effect chains and communication tasks. (Only in
        # the interconnected case.)
        self.interconnected = interconnected

        # Analysis results: (Are added during the analysis.)
        self.davare = 0  # Davare
        self.duerr_age = 0  # Duerr max data age
        self.duerr_react = 0  # Duerr max reaction time
        self.our_age = 0  # Our max data age
        self.our_react = 0  # Our max reaction time
        self.our_red_age = 0  # Our reduced max data age
        self.inter_our_age = 0  # Our max data age for interconn
        self.inter_our_red_age = 0  # Our reduced max data age for interconn
        self.inter_our_react = 0  # Our max reaction time for interconn
        self.kloda = 0  # Kloda
        self.becker = 0  # MRDA analysis from Becker

    def length(self):
        """Compute the length of a cause-effect chain."""
        return len(self.chain)

    @property
    def chain_disorder(self):
        """Compute the chain disorder. (Not explained in our paper.)

        The disorder of a chain is the number of priority inversions along
        the data propagation path.
        """
        return sum(
            1 if self.chain[i].priority > self.chain[i + 1].priority else 0
            for i in range(len(self.chain) - 1)
        )


class Task:
    """A task."""

    def __init__(self, task_id, task_phase, task_bcet, task_wcet, task_period,
                 task_deadline, priority=0, message=False):
        """Creates a task represented by ID, Phase, BCET, WCET, Period and
        Deadline.
        """
        self.id = str(task_id)
        self.phase = task_phase  # phase
        self.bcet = task_bcet  # best-case execution time
        self.wcet = task_wcet  # worst-case execution time
        self.period = task_period  # period
        self.deadline = task_deadline  # deadline
        self.priority = priority  # a lower value means higher priority
        self.message = message  # flag for communication tasks

        self.rt = task_wcet  # Worst-case response time, specified during analysis

    def __str__(self):
        """Print a task."""
        return (" Type: {type:^}\n ID: {id:^}\n Priority: {priority:^}\n"
                + " Phase: {phase:^} \n BCET: {bcet:^} \n WCET: {wcet:^} \n"
                + " Period: {period:^} \n Deadline: {deadline:^} \n"
                + " Response: {response:^}").format(
            type=str('Message') if self.message else str('Task'),
            id=self.id, priority=self.priority, phase=self.phase,
            bcet=self.bcet, wcet=self.wcet, period=self.period,
            deadline=self.deadline, response=self.rt)

    def copy(self):
        tsk = Task(self.id, self.phase, self.bcet, self.wcet,
                   self.period, self.deadline, self.priority, self.message)
        tsk.rt = self.rt
        return tsk


def compute_hyper(task_set):
        """Determine the hyperperiod of task_set."""
        # Collect periods.
        periods = []
        for task in task_set:
            if task.period not in periods:
                periods.append(task.period)
        # Compute least common multiple = hyperperiod.
        lcm = periods[0]
        for i in periods[1:]:
            lcm = int(lcm * i / math.gcd(lcm, i))
        return lcm

def davare_single(chain):  # Added in October 21
    """End-to-end latency analysis from Davare.

    Input: One chain.
    """
    latency = 0
    for task in chain.chain:
        latency += task.period + task.rt
    # Store result.
    chain.davare = latency

    return latency

class rel_dl_analyzer:
    def __init__(self):
        pass

    def rel(self, task, nmb):
        return task.phase + nmb * task.period

    def dl(self, task, nmb):
        return self.rel(task, nmb) + task.deadline

    def find_next_rel(self, task, bound):
        '''Find the index of the first job with release after the bound.'''
        for idx in itertools.count():
            if self.rel(task, idx) >= bound:
                return idx

    def find_prev_dl(self, task, bound):
        '''Find the index of the latest job with deadline before the bound.
        Note: returns -1 if no such job can be found.'''
        for idx, idx_next in zip(itertools.count(start=-1), itertools.count(start=0)):
            if not (self.dl(task, idx_next) <= bound):
                return idx

class re_we_analyzer():
    def __init__(self, bcet_schedule, wcet_schedule, hyperperiod):
        self.bc = bcet_schedule
        self.wc = wcet_schedule
        self.hyperperiod = hyperperiod

    def _get_entry(self, nmb, lst, tsk):
        '''get nmb-th entry of the list lst with task tsk.'''
        if nmb < 0:  # Case: out of range
            raise IndexError('nbm<0')
        # Case: index too high, has to be made smaller # TODO not sure if this is a good idea since the last entries could be wrong depending on the implementation of the scheduler ...
        elif nmb >= len(lst):
            # check one hyperperiod earlier
            # make new_nmb an integer value
            div, rem = divmod(self.hyperperiod, tsk.period)
            assert rem == 0
            new_nmb = nmb - div
            # add one hyperperiod
            # TODO this is not very efficient since we need the values several times.
            return [self.hyperperiod + entry for entry in self._get_entry(new_nmb, lst, tsk)]
        else:  # Case: entry can be used
            try:
                # print(f"lst: {lst}, nmb: {nmb}, tsk: {tsk}")
                return lst[nmb]
            except:
                breakpoint()

    def remin(self, task, nmb):
        '''returns the upper bound on read-event of the nbm-th job of a task.'''
        lst = self.bc[task]  # list that has the read-even minimum
        # choose read-event from list
        return self._get_entry(nmb, lst, task)[0]

    def remax(self, task, nmb):
        '''returns the upper bound on read-event of the nbm-th job of a task.'''
        lst = self.wc[task]  # list that has the read-even maximum
        # choose read-event from list
        return self._get_entry(nmb, lst, task)[0]

    def wemin(self, task, nmb):
        '''returns the upper bound on read-event of the nbm-th job of a task.'''
        lst = self.bc[task]  # list that has the write-even minimum
        # choose write-event from list
        return self._get_entry(nmb, lst, task)[1]

    def wemax(self, task, nmb):
        '''returns the upper bound on read-event of the nbm-th job of a task.'''
        lst = self.wc[task]  # list that has the write-even maximum
        # choose write-event from list
        return self._get_entry(nmb, lst, task)[1]

    def find_next_fw(self, curr_task_wc, nxt_task_bc, curr_index):
        '''Find next index for the abstract representation in forward manner.'''
        # wemax of current task
        curr_wemax = self.wemax(curr_task_wc, curr_index)
        curr_rel = curr_task_wc.phase + curr_index * \
            curr_task_wc.period  # release of current task

        for idx in itertools.count():
            idx_remin = self.remin(nxt_task_bc, idx)

            if (
                idx_remin >= curr_wemax  # first property
                # second property (lower priority value means higher priority!)
                or (curr_task_wc.priority < nxt_task_bc.priority and idx_remin >= curr_rel)
            ):
                return idx

    def len_abstr(self, abstr, last_tsk_wc, first_tsk_bc):
        '''Length of an abstract representation.'''
        # print(f"abstr: {abstr}, last_tsk_wc: {last_tsk_wc}, first_tsk_bc: {first_tsk_bc}")
        l = self.wemax(last_tsk_wc, abstr[-1])-self.remin(first_tsk_bc, abstr[0])
        print(f"wemax: {self.wemax(last_tsk_wc, abstr[-1])}, remin: {self.remin(first_tsk_bc, abstr[0])}")
        print(f"length: {l}")
        return self.wemax(last_tsk_wc, abstr[-1])-self.remin(first_tsk_bc, abstr[0])



def max_reac_local(chain, task_set_wcet, schedule_wcet, task_set_bcet, schedule_bcet):
    '''Main method for maximum reaction time.

    We construct all abstract represenations and compute the maximal length among them.
    - chain: cause-effect chain as list of tasks
    - task_set: the task set of the ECU that the ce chain lies on
    - schedule: the schedule of task_set (simulated beforehand)

    we distinguish between bcet and wcet task set and schedule.'''

    if chain.length() == 0:  # corner case
        return 0

    # Make analyzer
    ana = re_we_analyzer(schedule_bcet, schedule_wcet,
                         compute_hyper(task_set_wcet))
    # Chain of indeces that describes the cause-effect chain
    index_chain = [task_set_wcet.index(entry) for entry in chain.chain]

    # Set of all abstract representations
    all_abstr = []

    # useful values for break-condition
    hyper = compute_hyper(task_set_wcet)
    max_phase = max([task.phase for task in task_set_wcet])
    # print(f"max_phase: {max_phase}, hyper: {hyper}")
    for idx in itertools.count():
        # Compute idx-th abstract integer representation.
        abstr = []
        abstr.append(idx)  # first entry
        abstr.append(idx+1)  # second entry

        for idtsk, nxt_idtsk in zip(index_chain[:-1], index_chain[1:]):
            abstr.append(ana.find_next_fw(
                task_set_wcet[idtsk], task_set_bcet[nxt_idtsk], abstr[-1]))  # intermediate entries
            # print(f"abstr so far: {abstr}")
            # print(f"task_set_wcet[idtsk]: {task_set_wcet[idtsk]}, task_set_bcet[nxt_idtsk]: {task_set_bcet[nxt_idtsk]}, abstr[-1]: {abstr[-1]}")

        abstr.append(abstr[-1])  # last entry
        # print(f"abstr final: {abstr}")
        assert len(abstr) == chain.length() + 2

        all_abstr.append(abstr[:])

        # Break loop
        if (chain.chain[0].phase + idx * chain.chain[0].period) >= (max_phase + 2*hyper):
            break

        # print([task_set_wcet[i].priority for i in index_chain])

        # print([(schedule_bcet[task_set_bcet[i]][j][0], schedule_wcet[task_set_wcet[i]][j][1])
        #       for i, j in zip(index_chain, abstr[1:-1])])

        # breakpoint()

    # maximal length
    max_length = max([ana.len_abstr(abstr, task_set_wcet[index_chain[-1]],
                     task_set_bcet[index_chain[0]]) for abstr in all_abstr] + [0])
    chain.our_new_local_mrt = max_length
    return max_length


def execution_zero_schedule(task_set):
    '''Since the dispatcher can only handle execution time >0, we generate a "faked" schedule.'''
    hyperperiod = compute_hyper(task_set)
    max_phase = max([task.phase for task in task_set])

    # Initialize result dictionary.
    result = dict()
    for task in task_set:
        result[task] = []

    for task in task_set:
        curr_time = task.phase
        while curr_time <= max_phase + 2 * hyperperiod:
            # start and finish directly at release
            result[task].append((curr_time, curr_time))
            curr_time += task.period

    return result



def mrt_let(chain, task_set):
    '''Compute maximum reaction time when all tasks adhere to LET.
    This is an exact analysis.'''

    # Make analyzer
    ana = rel_dl_analyzer()

    # Set of forward chains
    fw = []

    # useful values for break-condition and valid check
    hyper = compute_hyper(task_set)
    max_phase = max([task.phase for task in task_set])
    max_first_read_event = max([ana.rel(task, 0) for task in task_set])

    for idx in itertools.count():
        # check valid
        if not (ana.rel(chain.chain[0], idx + 1) >= max_first_read_event):
            continue

        # Compute idx-th fw chain
        fwidx = []

        fwidx.append(ana.rel(chain.chain[0], idx))  # external activity
        fwidx.append(idx+1)  # first job

        for curr_task, nxt_task in zip(chain.chain[:-1], chain.chain[1:]):
            fwidx.append(  # add next release
                ana.find_next_rel(nxt_task, ana.dl(curr_task, fwidx[-1]))
            )

        fwidx.append(ana.dl(chain.chain[-1], fwidx[-1]))  # actuation

        assert len(fwidx) == chain.length() + 2

        fw.append(fwidx[:])

        # break condition
        if ana.rel(chain.chain[0], idx) >= (max_phase + 2*hyper):
            break

    max_length = max(fwidx[-1] - fwidx[0] for fwidx in fw)

    chain.mrt_let = max_length
    return max_length


def change_taskset_bcet(task_set, rat):
    """Copy task set and change the wcet/bcet of each task by a given ratio."""
    new_task_set = [task.copy() for task in task_set]
    for task in new_task_set:
        task.wcet = math.ceil(rat * task.wcet)
        task.bcet = math.ceil(rat * task.bcet)
    # Note: ceiling function makes sure there is never execution of 0
    return new_task_set


def schedule_task_set(ce_chains, task_set, print_status=False):
    """Return the schedule of some task_set.
    ce_chains is a list of ce_chains that will be computed later on.
    We need this to compute latency_upper_bound to determine the additional simulation time at the end.
    Note:
    - In case of error, None is returned.
    - E2E Davare has to be computed beforehand!"""

    try:
        # Preliminary: compute latency_upper_bound
        for ce in ce_chains:
            ce.davare = davare_single(ce)

        latency_upper_bound = max([ce.davare for ce in ce_chains])

        # Main part: Simulation part
        simulator = es.eventSimulator(task_set)

        # Determination of the variables used to compute the stop
        # condition of the simulation
        max_phase = max(task_set, key=lambda task: task.phase).phase
        max_period = max(task_set, key=lambda task: task.period).period
        hyper_period = a.Analyzer.determine_hyper_period(task_set)

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



def simple_experiment():
    task1 = Task(task_id="1", task_phase=2, task_bcet=1, task_wcet=3, task_period=5, task_deadline=5, priority=2)
    task2 = Task(task_id="2", task_phase=4, task_bcet=1, task_wcet=1, task_period=5, task_deadline=5, priority=1)

    task_set = [task2, task1]  
    
    chain = CauseEffectChain(1,task_set)
    ce_chains = [chain]

    # wcet_schedule = generate_simple_schedule(task_set, use_bcet=False, scale=1.0)
    # bcet_schedule = generate_simple_schedule(task_set, use_bcet=True, scale=1.0)
    wcet_schedule = schedule_task_set(ce_chains, task_set, print_status=False)
    bcet_schedule = schedule_task_set(ce_chains, task_set, print_status=False)
    
    print(f"WCET: {wcet_schedule}")
    print(f"BCET: {bcet_schedule}")

    try:
        mrt_result = mrt_let(chain, task_set)
        print(f"LET: {mrt_result}")
    except Exception as e:
        print(f"LET error: {e}")

    try:        
        max_reac_result = max_reac_local(chain, task_set, wcet_schedule, task_set, bcet_schedule)
        print(f"imp: {max_reac_result}")
    except Exception as e:
        print(f"error: {e}")



if __name__ == "__main__":
    simple_experiment()