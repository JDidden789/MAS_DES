"""
DES of the MAS. This DES is used to optimize certain aspects of the
MAS such as the bids. It can be used to quickly run multiple experiments.
The results can then be implemented in the MAS to see the effects.
"""
import csv
import itertools
import random
from collections import defaultdict
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
import matplotlib.pyplot as plt

import numpy as np
# import matplotlib.cbook
import pandas as pd
import simpy
from simpy import *

# warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

# General Settings
number = 2500  # Max number of jobs if infinite is false
noJobCap = True  # For infinite
maxTime = 10000.0  # Runtime limit
# Machine shop settings
processingTimes = [[6.75, 3.75, 2.5, 7.5], [3.75, 5.0, 7.5], [3.75, 2.5, 8.75, 5.0, 5.0]]
operationOrder = [[3, 1, 2, 5], [4, 1, 3], [2, 5, 1, 4, 3]]
numberOfOperations = [4, 3, 5]
machinesPerWC = [4, 2, 5, 3, 2]
machine_number_WC = [[1, 2, 3, 4], [5, 6], [7, 8, 9, 10, 11], [12, 13, 14], [15, 16]]
setupTime = [[0, 0.625, 1.25], [0.625, 0, 0.8], [1.25, 0.8, 0]]
mean_setup = [0.515, 0.306, 0.515, 0.429, 0.306]
demand = [0.2, 0.5, 0.3]
noOfWC = range(len(machinesPerWC))

if noJobCap:
    number = 0

"Initial parameters of the GES"
noAttributes = 8
noAttributesJob = 4
totalAttributes = noAttributes + noAttributesJob

no_generation = 500


def list_duplicates(seq):
    tally = defaultdict(list)
    for ii, item in enumerate(seq):
        tally[item].append(ii)
    return ((key, locs) for key, locs in tally.items()
            if len(locs) >= 1)


def job_init(choice, job):
    if choice == 0:
        index_job = 0
    elif choice == 1:
        priority = []
        for ii, j in enumerate(job):
            priority.append(j.processingTime[j.currentOperation - 1] / j.priority)
        index_job = np.argmin(priority)
    elif choice == 2:
        priority = []
        for ii, j in enumerate(job):
            priority.append(remain_processing_time(j) / j.priority)
        index_job = np.argmin(priority)
    elif choice == 3:
        priority = []
        for ii, j in enumerate(job):
            priority.append(j.dueDate[j.numberOfOperations] / j.priority)
        index_job = np.argmin(priority)
    else:
        priority = []
        for ii, j in enumerate(job):
            priority.append(j.dueDate[j.currentOperation] / j.priority)
        index_job = np.argmin(priority)

    return index_job


def bid_calulculation_other(pool, noMachines, meanSetup, meanProcessingTime, meanDueDate, maxDueDate, minDueDate,
                            due_date, processingTime, current_time, setup_time, priority, r_proc):
    if meanSetup == 0:
        meanSetup = 0.01

    mu = pool / noMachines
    eta = meanSetup / meanProcessingTime
    if mu > 5:
        beta = 0.4 + 10 / (mu ** 2) - eta / 7
    else:
        beta = 0.4 + 10 / (mu ** 2) - eta / 7
    tau = 1 - meanDueDate / ((beta * meanSetup + meanProcessingTime) * mu)
    R = (maxDueDate - minDueDate) / ((beta * meanSetup + meanProcessingTime) * mu)

    if (tau < 0.5) | ((eta < 0.5) & (mu > 5)):
        k_1 = 1.2 * np.log(mu) - R - 0.5
    else:
        k_1 = 1.2 * np.log(mu) - R

    if tau < 0.8:
        k_2 = tau / (1.8 * np.sqrt(eta))
    else:
        k_2 = tau / (2 * np.sqrt(eta))

    k_1 = 2

    # print(mu)
    bid = priority / processingTime * np.exp(
        -max(0, (due_date - r_proc - current_time)) / (k_1 * meanProcessingTime)) * np.exp(
        -setup_time / (k_2 * meanSetup))

    # if bid > 10 ** 9:
    #     print(eta, mu, tau)

    return bid


def job_seq(job, env, k, choice, setuptime):
    priority = []
    meanProcTime = np.mean([jj.processingTime[jj.currentOperation - 1] for jj in job])
    meanDueDate = np.mean([[jj.dueDate[jj.numberOfOperations]] for jj in job])
    maxDueDate = max([jj.dueDate[jj.numberOfOperations] for jj in job])
    minDueDate = min([jj.dueDate[jj.numberOfOperations] for jj in job])
    if choice == 0:
        # FCFS
        index_job = 0
    else:

        for ii, j in enumerate(job):
            setup_time = setuptime[ii]
            # WSPT
            if choice == 1:
                priority.append((j.processingTime[j.currentOperation - 1] + setup_time) / j.priority)
            elif choice == 2:
                priority.append(remain_processing_time(j) / j.priority)
            # WEDD
            elif choice == 3:
                priority.append(j.dueDate[j.numberOfOperations] / j.priority)
            # WODD
            elif choice == 4:
                priority.append(j.dueDate[j.currentOperation] / j.priority)
            # WCR
            elif choice == 5:
                priority_new = (j.dueDate[j.numberOfOperations] - j.processingTime[
                    j.currentOperation - 1] - setup_time - env.now) / (
                                   remain_processing_time(j))
                if priority_new > 0:
                    priority.append(priority_new / j.priority)
                else:
                    priority.append(priority_new * j.priority)
            # SLACK
            elif choice == 6:
                priority_new = (j.dueDate[j.currentOperation] - j.processingTime[
                    j.currentOperation - 1] - setup_time - env.now)
                if priority_new > 0:
                    priority.append(priority_new / j.priority)
                else:
                    priority.append(priority_new * j.priority)
            # WMDD
            elif choice == 7:
                priority.append(1 / j.priority * max(j.processingTime[j.currentOperation - 1] + setup_time,
                                                     j.dueDate[j.numberOfOperations] - env.now))
            # WCOVERT
            elif choice == 8:
                c_i = j.priority / j.processingTime[j.currentOperation - 1] * max(0, 1 - max(0, j.dueDate[
                    j.numberOfOperations] - env.now - remain_processing_time(j)) / (k * remain_processing_time(j) -
                                                                                    j.processingTime[
                                                                                        j.currentOperation - 1] - setup_time))
                priority.append(c_i)
            else:
                prio = bid_calulculation_other(len(job), 1, np.mean(setuptime), meanProcTime, meanDueDate, maxDueDate,
                                               minDueDate, j.dueDate[j.numberOfOperations], j.processingTime[
                                                   j.currentOperation - 1], env.now, setup_time, j.priority,
                                               remain_processing_time(j))
                priority.append(prio)

        if (choice == 8) | (choice == 9):
            index_job = np.argmax(priority)
        else:
            index_job = np.argmin(priority)

    return index_job


# pool, noMachines, meanSetup, meanProcessingTime, meanDueDate, maxDueDate, minDueDate,
#                             due_date, processingTime, current_time, setup_time, priority

def machine_choice(env, job, noOfMachines, currentWC, job_shop, machine, store, mac_choice, job_choice):
    machine_index = []
    for jj in range(noOfMachines):
        if mac_choice == 0:
            machine_index.append(len(machine[(jj, currentWC - 1)].items))
        else:
            machine_index.append(expected_start_time(machine[(jj, currentWC - 1)]))
    chosen_machine = np.argmin(machine_index)
    index_job = job_init(job_choice, job)
    # Append to machine
    put_job_in_queue(currentWC, chosen_machine, job[index_job], job_shop, env, machine)

    yield store.get(lambda mm: mm == job[index_job])


def expected_start_time(machine):
    extra_start_time = 0
    # print(machine)
    for ii, j in enumerate(machine.items):
        if ii == 0:
            extra_start_time = j.processingTime[j.currentOperation - 1]
            last_job = j
        else:
            extra_start_time += (j.processingTime[j.currentOperation - 1] + setupTime[j.type - 1][last_job.type - 1])
            last_job = j

    return extra_start_time


def remain_processing_time(job):
    total_rp = 0
    # total_rp = sum((job.processingTime[ii]) for ii in range(job.currentOperation - 1, job.numberOfOperations))
    for ii in range(job.currentOperation - 1, job.numberOfOperations):
        total_rp += job.processingTime[ii]

    return total_rp


def next_workstation(job, job_shop, env, min_job, max_job, max_wip):
    if job.currentOperation + 1 <= job.numberOfOperations:
        job.currentOperation += 1
        nextWC = operationOrder[job.type - 1][job.currentOperation - 1]
        store = job_shop.storeWC[nextWC - 1]
        store.put(job)
    else:
        finish_time = env.now
        job_shop.totalWIP.append(job_shop.WIP)
        job_shop.tardiness[job.number] = max(job.priority * (finish_time - job.dueDate[job.numberOfOperations]), 0)
        job_shop.priority[job.number] = job.priority
        job_shop.WIP -= 1
        job_shop.flowtime[job.number] = finish_time - job.dueDate[0]
        # finished_job += 1
        if job.number > max_job:
            if np.count_nonzero(job_shop.flowtime[min_job:max_job]) == 2000:
                job_shop.finish_time = env.now
                job_shop.end_event.succeed()

        if env.now > 20_000:
            job_shop.end_event.succeed()
            job_shop.early_termination = 1
            job_shop.finish_time = env.now


def normalize(value, max_value, min_value):
    return (value - min_value) / (max_value - min_value)


def expected_setup_time(new_job, job_shop, list_jobs):
    set_time = []

    for f in list_jobs:
        set_time.append(setupTime[f.type - 1][new_job.type - 1])

    return max(set_time)


def set_makespan(current_makespan, job, last_job, env, setup_time):
    add = current_makespan + job.processingTime[job.currentOperation - 1] + setup_time

    new = env.now + job.processingTime[job.currentOperation - 1] + setup_time

    return max(add, new)


def put_job_in_queue(currentWC, choice, job, job_shop, env, machines):
    machines[(choice, currentWC - 1)].put(job)
    if not job_shop.condition_flag[(choice, currentWC - 1)].triggered:
        job_shop.condition_flag[(choice, currentWC - 1)].succeed()


def machine_processing(job_shop, current_WC, machine_number, env, last_job, machine,
                       makespan, seq_choice, k, min_job, max_job, max_wip):
    while True:
        relative_machine = machine_number_WC[current_WC - 1].index(machine_number)
        if machine.items:
            setup_time = []
            # priority_list = []
            if not last_job[relative_machine]:
                setup_time.append(0)
            else:
                for job in machine.items:
                    setuptime = setupTime[job.type - 1][int(last_job[relative_machine]) - 1]
                    setup_time.append(setuptime)
            ind_processing_job = job_seq(machine.items, env, k, seq_choice, setup_time)

            next_job = machine.items[ind_processing_job]
            setuptime = setup_time[ind_processing_job]
            tip = next_job.processingTime[next_job.currentOperation - 1] + setuptime
            makespan[relative_machine] = set_makespan(makespan[relative_machine], next_job, last_job[relative_machine],
                                                      env, setuptime)
            job_shop.utilization[(relative_machine, current_WC - 1)] = job_shop.utilization[(
                relative_machine, current_WC - 1)] + setuptime + next_job.processingTime[next_job.currentOperation - 1]
            last_job[relative_machine] = next_job.type
            machine.items.remove(next_job)
            yield env.timeout(tip)
            next_workstation(next_job, job_shop, env, min_job, max_job, max_wip)
        else:
            yield job_shop.condition_flag[(relative_machine, current_WC - 1)]
            job_shop.condition_flag[(relative_machine, current_WC - 1)] = simpy.Event(env)


def cfp_wc(env, machine, store, job_shop, currentWC, mac_choice, job_choice):
    while True:
        if store.items:
            job_shop.QueuesWC[currentWC].append(
                {i: len(job_shop.machine_per_wc[(i, currentWC)].items) for i in range(machinesPerWC[currentWC])})
            c = machine_choice(env, store.items, machinesPerWC[currentWC], currentWC + 1, job_shop,
                               machine, store, mac_choice, job_choice)
            env.process(c)
        tib = 0.5
        yield env.timeout(tib)


def source(env, number1, interval, job_shop, due_date_setting, min_job):
    while True:  # Needed for infinite case as True refers to "until".
        ii = number1
        if ii == min_job:
            job_shop.start_time = env.now
        number1 += 1
        job = New_Job('job%02d' % ii, env, ii, due_date_setting)
        job_shop.tardiness.append(-1)
        job_shop.priority.append(0)
        job_shop.flowtime.append(0)
        job_shop.WIP += 1
        firstWC = operationOrder[job.type - 1][0]
        store = job_shop.storeWC[firstWC - 1]
        store.put(job)
        tib = random.expovariate(1.0 / interval)
        yield env.timeout(tib)


class jobShop:
    def __init__(self, env):
        self.machine_per_wc = {(ii, jj): Store(env) for jj in noOfWC for ii in range(machinesPerWC[jj])}
        self.storeWC = {ii: FilterStore(env) for ii in noOfWC}
        self.QueuesWC = {jj: [] for jj in noOfWC}
        self.scheduleWC = {ii: [] for ii in noOfWC}
        self.makespanWC = {ii: np.zeros(machinesPerWC[ii]) for ii in noOfWC}
        self.last_job_WC = {ii: np.zeros(machinesPerWC[ii]) for ii in noOfWC}
        self.condition_flag = {(ii, jj): simpy.Event(env) for jj in noOfWC for ii in range(machinesPerWC[jj])}

        self.flowtime = []
        self.tardiness = []
        self.WIP = 0
        self.early_termination = 0
        self.utilization = {(ii, jj): 0 for jj in noOfWC for ii in range(machinesPerWC[jj])}
        self.finish_time = 0
        self.totalWIP = []
        self.priority = []

        self.bids = []
        self.start_time = 0


class New_Job:
    def __init__(self, name, env, number1, dueDateTightness):
        jobType = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2], k=1)
        jobWeight = random.choices([1, 3, 10], weights=[0.5, 0.3, 0.2], k=1)
        self.type = jobType[0]
        self.priority = jobWeight[0]
        self.number = number1
        self.name = name
        self.currentOperation = 1
        self.processingTime = np.zeros(numberOfOperations[self.type - 1])
        self.dueDate = np.zeros(numberOfOperations[self.type - 1] + 1)
        self.dueDate[0] = env.now
        self.operationOrder = operationOrder[self.type - 1]
        self.numberOfOperations = numberOfOperations[self.type - 1]
        for ii in range(self.numberOfOperations):
            meanPT = processingTimes[self.type - 1][ii]
            self.processingTime[ii] = meanPT
            self.dueDate[ii + 1] = self.dueDate[ii] + self.processingTime[ii] * dueDateTightness


def get_objectives(job_shop, min_job, max_job, early_termination):
    """This function gathers numerous results from a simulation run"""
    no_tardy_jobs_p1 = 0
    no_tardy_jobs_p2 = 0
    no_tardy_jobs_p3 = 0
    total_p1 = 0
    total_p2 = 0
    total_p3 = 0
    early_term = 0
    if early_termination == 1:
        early_term += 1
        makespan = job_shop.finish_time - job_shop.start_time
        flow_time = np.nanmean(job_shop.flowtime[min_job:max_job]) + 10_000 - np.count_nonzero(
            job_shop.flowtime[min_job:max_job])
        mean_tardiness = np.nanmean(np.nonzero(job_shop.tardiness[min_job:max_job])) + 10_000 - np.count_nonzero(
            job_shop.flowtime[min_job:max_job])
        max_tardiness = np.nanmax(job_shop.tardiness[min_job:max_job])
        for ii in range(min_job, len(job_shop.tardiness)):
            if job_shop.priority[ii] == 1:
                if job_shop.tardiness[ii] > 0:
                    no_tardy_jobs_p1 += 1
                total_p1 += 1
            elif job_shop.priority[ii] == 3:
                if job_shop.tardiness[ii] > 0:
                    no_tardy_jobs_p2 += 1
                total_p2 += 1
            elif job_shop.priority[ii] == 10:
                if job_shop.tardiness[ii] > 0:
                    no_tardy_jobs_p3 += 1
                total_p3 += 1
        # WIP Level
        mean_WIP = np.mean(job_shop.totalWIP)
    else:
        makespan = job_shop.finish_time - job_shop.start_time
        # Mean Flow Time
        flow_time = np.nanmean(job_shop.flowtime[min_job:max_job])
        # Mean Tardiness
        mean_tardiness = np.nanmean(job_shop.tardiness[min_job:max_job])
        # Max Tardiness
        max_tardiness = max(job_shop.tardiness[min_job:max_job])
        # print(len(job_shop.priority))
        # No of Tardy Jobs
        for ii in range(min_job, max_job):
            if job_shop.priority[ii] == 1:
                if job_shop.tardiness[ii] > 0:
                    no_tardy_jobs_p1 += 1
                total_p1 += 1
            elif job_shop.priority[ii] == 3:
                if job_shop.tardiness[ii] > 0:
                    no_tardy_jobs_p2 += 1
                total_p2 += 1
            elif job_shop.priority[ii] == 10:
                if job_shop.tardiness[ii] > 0:
                    no_tardy_jobs_p3 += 1
                total_p3 += 1
        # WIP Level
        mean_WIP = np.mean(job_shop.totalWIP)

    # print(makespan)

    return makespan, flow_time, mean_tardiness, max_tardiness, no_tardy_jobs_p1 / total_p1, no_tardy_jobs_p2 / total_p2, no_tardy_jobs_p3 / total_p2, mean_WIP, early_term


def do_simulation_with_weights(arrivalMean, due_date_tightness, mac_rule, job_rule, seq_rule, min_job, max_job, max_wip,
                               iter):
    # print(mean_weight_new)
    random.seed(iter)

    env = Environment()
    job_shop = jobShop(env)
    env.process(source(env, 0, arrivalMean, job_shop, due_date_tightness, min_job))

    for wc in range(len(machinesPerWC)):
        last_job = job_shop.last_job_WC[wc]
        makespanWC = job_shop.makespanWC[wc]
        store = job_shop.storeWC[wc]

        env.process(cfp_wc(env, job_shop.machine_per_wc, store, job_shop, wc, mac_rule, job_rule))

        for ii in range(machinesPerWC[wc]):
            machine = job_shop.machine_per_wc[(ii, wc)]
            utilization = job_shop.utilization[(ii, wc)]

            env.process(
                machine_processing(job_shop, wc + 1, machine_number_WC[wc][ii], env, last_job,
                                   machine, makespanWC, seq_rule, due_date_tightness, min_job, max_job, max_wip))

    job_shop.end_event = env.event()

    env.run(until=job_shop.end_event)

    # no_tardy_jobs_p1 = 0
    # no_tardy_jobs_p2 = 0
    # no_tardy_jobs_p3 = 0
    # total_p1 = 0
    # total_p2 = 0
    # total_p3 = 0
    # # Makespan
    # makespan = job_shop.finish_time - job_shop.start_time
    # # Mean Flow Time
    # flow_time = np.nanmean(job_shop.flowtime[min_job:max_job])
    # # Mean Tardiness
    # mean_tardiness = np.nanmean(job_shop.tardiness[min_job:max_job])
    # # Max Tardiness
    # max_tardiness = max(job_shop.tardiness[min_job:max_job])
    # # print(len(job_shop.priority))
    # # No of Tardy Jobs
    # for i in range(min_job, max_job):
    #     if job_shop.priority[i] == 1:
    #         if job_shop.tardiness[i] > 0:
    #             no_tardy_jobs_p1 += 1
    #         total_p1 += 1
    #     elif job_shop.priority[i] == 3:
    #         if job_shop.tardiness[i] > 0:
    #             no_tardy_jobs_p2 += 1
    #         total_p2 += 1
    #     elif job_shop.priority[i] == 10:
    #         if job_shop.tardiness[i] > 0:
    #             no_tardy_jobs_p3 += 1
    #         total_p3 += 1
    # # WIP Level
    # mean_WIP = np.mean(job_shop.totalWIP)

    makespan, flow_time, mean_tardiness, max_tardiness, no_tardy_jobs_p1, no_tardy_jobs_p2, no_tardy_jobs_p3, mean_WIP, early_term = get_objectives(
        job_shop, min_job, max_job, job_shop.early_termination)  # Gather all results

    return makespan, flow_time, mean_tardiness, max_tardiness, no_tardy_jobs_p1, no_tardy_jobs_p2, no_tardy_jobs_p3, mean_WIP, early_term


if __name__ == '__main__':
    # df = pd.read_csv('Runs/Attribute_Runs/Run-Weights-NoDD-85-4-5000.csv', header=None)
    # weights = df.values.tolist()
    #

    no_machine_rules = 2
    no_job_rules = 5
    no_seq_rules = 10

    final_obj = []
    final_std = []

    no_runs = 50
    no_processes = 25
    # final_result = np.zeros((no_runs, 8))
    # results = []
    arrival_time = [1.5429, 1.5429, 1.4572, 1.4572, 1.3804, 1.3804]
    utilization = [85, 85, 90, 90, 95, 95]
    due_date_settings = [4, 6, 4, 6, 4, 6]
    min_jobs = [499, 499, 999, 999, 1499, 1499]
    max_jobs = [2499, 2499, 2999, 2999, 3499, 3499]
    wip_max = [150, 150, 200, 200, 300, 300]

    for i in range(len(arrival_time)):
        final_result = np.zeros((no_runs, 8))
        results = []
        print(i)
        for (f, e, d) in itertools.product(range(no_seq_rules), range(no_job_rules), range(no_machine_rules)):
            print(d, e, f)
            obj = np.zeros(no_runs)
            for j in range(int(no_runs / no_processes)):
                jobshop_pool = Pool(processes=no_processes)
                seeds = range(j * no_processes, j * no_processes + no_processes)
                func1 = partial(do_simulation_with_weights, arrival_time[i], due_date_settings[i], d, e, f, min_jobs[i],
                                max_jobs[i], wip_max[i])
                makespan_per_seed = jobshop_pool.map(func1, seeds)
                # print(makespan_per_seed)
                for h, o in itertools.product(range(no_processes), range(8)):
                    final_result[h + j * no_processes][o] = makespan_per_seed[h][o]
            results.append(list(np.mean(final_result, axis=0)))
            print(results)

        results = pd.DataFrame(results,
                               columns=['Makespan', 'Mean Flow Time', 'Mean Weighted Tardiness',
                                        'Max Weighted Tardiness',
                                        'No. Tardy Jobs P1', 'No. Tardy Jobs P2', 'No. Tardy Jobs P3', 'Mean WIP'])
        file_name = f"Results/Dispatching-{utilization[i]}-{due_date_settings[i]}-1.csv"
        results.to_csv(file_name)

    # with open(filename2, 'w') as file2:
    #     writer = csv.writer(file2)
    #     writer.writerows(results)

    # arrival_time = [1.5429, 1.5429, 1.5429, 1.4572, 1.4572, 1.4572, 1.3804, 1.3804, 1.3804]
