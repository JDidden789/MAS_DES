"""
DES of the MAS. This DES is used to optimize certain aspects of the
MAS such as the bids. It can be used to quickly run multiple experiments.
The results can then be implemented in the MAS to see the effects.
"""
import csv
import random
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

import numpy as np
# import matplotlib.cbook
import pandas
import pandas as pd
import simpy
from simpy import *

# warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

number = 2500  # Max number of jobs if infinite is false
noJobCap = True  # For infinite
maxTime = 10000.0  # Runtime limit
# Old job settings
# processingTimes = [[6.75, 3.75, 2.5, 7.5], [3.75, 5.0, 7.5], [3.75, 2.5, 8.75, 5.0, 5.0]]
# operationOrder = [[3, 1, 2, 5], [4, 1, 3], [2, 5, 1, 4, 3]]
# numberOfOperations = [4, 3, 5]
# machinesPerWC = [4, 2, 5, 3, 2]
# machine_number_WC = [[1, 2, 3, 4], [5, 6], [7, 8, 9, 10, 11], [12, 13, 14], [15, 16]]
# setupTime = [[0, 0.625, 1.25], [0.625, 0, 0.8], [1.25, 0.8, 0]]
# mean_setup = [0.515, 0.306, 0.515, 0.429, 0.306]

# New Job Settings
processingTimes = [[50, 40, 20, 40, 30, 65, 25, 70],
                   [70, 20, 30, 60, 70, 60, 40, 50],
                   [35, 60, 20, 35, 40, 20, 25, 80],
                   [30, 60, 30, 70, 30, 40, 70, 30]]
operationOrder = [[1, 2, 4, 3, 2, 1, 3, 4], [1, 2, 4, 3, 2, 1, 3, 4], [1, 2, 4, 3, 2, 1, 3, 4],
                  [1, 2, 4, 3, 2, 1, 3, 4]]
numberOfOperations = [8, 8, 8, 8]
machinesPerWC = [2, 2, 2, 2]
machine_number_WC = [[1, 2], [3, 4], [5, 6], [7, 8]]
setupTime = [[0, 0.1, 0.1, 0.1],
             [0.1, 0, 0.1, 0.1],
             [0.1, 0.1, 0, 0.1],
             [0.1, 0.1, 0.1, 0]]
mean_setup = [0.1, 0.1, 0.1, 0.1]
demand = [0.25, 0.25, 0.25, 0.25]

# arrivalMean = 1.4572
# Mean of arrival process
# dueDateTightness = 3

if noJobCap:
    number = 0

"Initial parameters of the GES"
noAttributes = 7
noAttributesJob = 3
totalAttributes = noAttributes

no_generation = 500


def list_duplicates(seq):
    tally = defaultdict(list)
    for ii, item in enumerate(seq):
        tally[item].append(ii)
    return ((key, locs) for key, locs in tally.items()
            if len(locs) >= 1)


def bid_winner(env, job, noOfMachines, currentWC, job_shop, last_job, makespan_currentWC, machine, store):
    # test_weights_new = job_shop.test_weights
    current_bid = [0] * noOfMachines
    current_job = [0] * noOfMachines
    best_bid = []
    best_job = []
    no_of_jobs = len(job)
    # bid_structure = []
    # removed_job = []
    # last_job = eval('job_shop.last_job_WC' + str(currentWC))
    # makespan_currentWC = eval('job_shop.makespanWC' + str(currentWC))
    # machine = eval('job_shop.machinesWC' + str(currentWC))

    # meanProcessingTime = 0

    # dueDateVector = np.zeros(no_jobs)
    # setup_time_mean = np.zeros(noOfMachines)
    # for ii in range(no_jobs):
    #     dueDateVector[ii] = job[ii].dueDate[job[ii].currentOperation]
    #     meanProcessingTime += job[ii].processingTime[job[ii].currentOperation - 1]
    #     for jj in range(noOfMachines):
    #         if last_job[jj] != 0:
    #             setup_time_mean[jj] += setupTime[job[ii].type - 1][int(last_job[jj]) - 1]
    #         else:
    #             setup_time_mean[jj] += 0.01
    #
    # meanDueDate = np.mean(dueDateVector)
    # maxDueDate = max(dueDateVector)
    # minDueDate = min(dueDateVector)
    # setup_time_mean = setup_time_mean / no_jobs
    # meanProcessingTime = meanProcessingTime / no_jobs
    total_rp = [0] * no_of_jobs
    for j in range(no_of_jobs):
        total_rp[j] = (remain_processing_time(job[j]))

    for jj in range(noOfMachines):
        expected_start = expected_start_time(jj, currentWC, machine)
        start_time = max(env.now, makespan_currentWC[jj] + expected_start)
        new_bid = [0] * no_of_jobs
        pool = len(machine[jj].items)
        i = 0
        for j in job:
            setup_time = setupTime[j.type - 1][int(last_job[jj]) - 1]
            attributes = bid_calculation(job_shop.test_weights, machine_number_WC[currentWC - 1][jj],
                                         j.processingTime[j.currentOperation - 1], start_time, j.currentOperation,
                                         j.numberOfOperations,
                                         j.dueDate[j.currentOperation], total_rp[i], expected_start,
                                         j.dueDate[j.numberOfOperations], env.now,
                                         j.priority, setup_time, pool)
            new_bid[i] = sum(attributes)
            # if j.number > 499:
            #     bid_structure = [currentWC, jj, new_bid[i], j.name + str(j.currentOperation), env.now]
            #     bid_structure.extend(attributes)
            #     job_shop.bids.append(bid_structure)
            i += 1

        ind_winning_job = new_bid.index(max(new_bid))
        current_bid[jj] = new_bid[ind_winning_job]
        current_job[jj] = ind_winning_job

    # Determine the winning bids
    sorted_list = sorted(list_duplicates(current_job))
    for dup in sorted_list:
        bestmachine = dup[1][0]
        bestbid = current_bid[bestmachine]

        for ii in dup[1]:
            if bestbid <= current_bid[ii]:
                bestbid = current_bid[ii]
                bestmachine = ii

        best_bid.append(bestmachine)  # Machine winner
        best_job.append(int(dup[0]))  # Job to be processed
        # if job[int(dup[0])].number > 499:
        #     job_shop.winningbids.append(
        #         [currentWC, bestmachine, job[int(dup[0])].name + str(job[int(dup[0])].currentOperation),
        #          job[int(dup[0])].type, job[int(dup[0])].priority])

    for ii in range(len(best_job)):
        put_job_in_queue(currentWC, best_bid[ii], job[best_job[ii]], job_shop, env, machine)

    for ii in reversed(best_job):
        yield store.get(lambda mm: mm == job[ii])


def expected_start_time(jj, currentWC, machine):
    # machine = eval('job_shop.machinesWC' + str(currentWC))
    extra_start_time = 0
    for kk in range(len(machine[jj].items)):
        current_job = machine[jj].items[kk]
        extra_start_time += (current_job.processingTime[current_job.currentOperation - 1]/60 + mean_setup[currentWC - 1])

    return extra_start_time


def bid_calulculation_other(pool, noMachines, meanSetup, meanProcessingTime, meanDueDate, maxDueDate, minDueDate,
                            due_date, processingTime, current_time, setup_time, job, priority_value):
    if meanSetup == 0:
        meanSetup = 0.01

    mu = pool / noMachines
    eta = meanSetup / meanProcessingTime
    beta = 0.4 - 10 / (mu ** 2) - eta / 7
    tau = 1 - meanDueDate / (beta * meanSetup + meanProcessingTime) * mu
    R = (maxDueDate - minDueDate) / (beta * meanSetup + meanProcessingTime) * mu

    if (tau < 0.5) | ((eta < 0.5) & (mu > 5)):
        k_1 = 1.2 * np.log(mu) - R - 0.5
    else:
        k_1 = 1.2 * np.log(mu) - R

    if tau < 0.8:
        k_2 = tau / (1.8 * np.sqrt(eta))
    else:
        k_2 = tau / (2 * np.sqrt(eta))

    bid = priority_value / processingTime * np.exp(
        -max(0, (due_date - processingTime - setup_time - current_time)) / (k_1 * meanProcessingTime)) * np.exp(
        -setup_time / (k_2 * meanSetup))

    # if bid > 10 ** 9:
    #     print(eta, mu, tau)

    return bid


def remain_processing_time(job):
    total_rp = 0
    for ii in range(job.currentOperation - 1, job.numberOfOperations):
        total_rp += job.processingTime[ii] / 60

    return total_rp


def next_workstation(job, job_shop, env, all_store):
    if job.currentOperation + 1 <= job.numberOfOperations:
        job.currentOperation += 1
        nextWC = operationOrder[job.type - 1][job.currentOperation - 1]
        store = all_store[nextWC - 1]
        store.put(job)
    else:
        finish_time = env.now
        job_shop.tardiness[job.number] = max(job.priority * (finish_time - job.dueDate[job.numberOfOperations]), 0)
        job_shop.WIP -= 1
        job_shop.makespan[job.number] = finish_time - job.dueDate[0]

        if job.number > 2499:
            if np.count_nonzero(job_shop.makespan[499:2499]) == 2000:
                job_shop.finishtime = env.now
                job_shop.end_event.succeed()

        # if (job_shop.WIP > 2500) | (env.now > 30_000):
        #     # print(job_shop.WIP)
        #     job_shop.end_event.succeed()
        #     job_shop.early_termination = 1


def bid_calculation(weights_new, machinenumber, processing_time, start_time,
                    current, total, due_date_operation, total_rp, expected_start, due_date, now, job_priority, setup, pool):
    attribute = [0] * noAttributes
    # print(machinenumber)
    attribute[0] = processing_time / 60 / 1.33 * weights_new[machinenumber - 1][0]
    attribute[1] = (current) / 8 * weights_new[machinenumber - 1][1]
    attribute[2] = (due_date - start_time + 100) / (26.67 + 100) * \
                   weights_new[machinenumber - 1][2]
    attribute[3] = total_rp / 6.67 * weights_new[machinenumber - 1][3]
    attribute[4] = (((due_date - now) / total_rp) + 100) / (4 + 100) * weights_new[machinenumber - 1][4]  # Critical Ratio
    # attribute[5] = (job_priority - 1) / (10 - 1) * weights_new[machinenumber - 1][5]  # Job Weight
    attribute[5] = pool / 100 * weights_new[machinenumber - 1][5]  # Current Workload of Machine
    attribute[6] = setup / 0.1 * weights_new[machinenumber - 1][6]  # Current Workload of Machine

    return attribute


def normalize(value, max_value, min_value):
    return (value - min_value) / (max_value - min_value)


def expected_setup_time(new_job, job_shop, list_jobs):
    priority = []

    for f in list_jobs:
        priority.append(setupTime[f.type - 1][new_job.type - 1])

    return max(priority)


def set_makespan(current_makespan, job, last_job, env, setup_time):
    # if last_job != 0:
    #     setup_time = setupTime[job.type - 1][int(last_job) - 1]
    # else:
    #     setup_time = 0
    add = current_makespan + job.processingTime[job.currentOperation - 1] / 60 + setup_time

    new = env.now + job.processingTime[job.currentOperation - 1] / 60 + setup_time

    return max(add, new)


def put_job_in_queue(currentWC, choice, job, job_shop, env, machines):
    machines[choice].put(job)
    if not job_shop.condition_flag[currentWC - 1][choice].triggered:
        job_shop.condition_flag[currentWC - 1][choice].succeed()


def choose_job_queue(weights_new_job, machinenumber, processing_time, due_date, env, setup_time, job_priority):
    attribute_job = [0] * noAttributesJob
    attribute_job[2] = setup_time / 0.5 * weights_new_job[machinenumber - 1][noAttributes + 2]
    attribute_job[1] = (job_priority - 1) / (10 - 1) * weights_new_job[machinenumber - 1][noAttributes + 1]
    attribute_job[0] = (due_date - processing_time - setup_time - env.now - (-400)) / (100 + 400) * \
                       weights_new_job[machinenumber - 1][noAttributes]

    # attribute_job[2] = normalize(setup_time, 1.25, 0) * weights_new_job[machinenumber - 1][noAttributes + 2]
    # attribute_job[1] = normalize(job_priority, 10, 1) * weights_new_job[machinenumber - 1][noAttributes + 1]
    # attribute_job[0] = normalize(due_date - processing_time - setup_time - env.now, 100, -400) * \
    #                    weights_new_job[machinenumber - 1][noAttributes]

    # total_bid = sum(attribute_job)
    return sum(attribute_job)


def machine_processing(job_shop, current_WC, machine_number, env, weights_new, relative_machine, last_job, machine,
                       makespan, all_store):
    while True:
        relative_machine = machine_number_WC[current_WC - 1].index(machine_number)
        # print( machine_number)
        if machine[relative_machine].items:
            # setup_time = []
            # priority_list = []
            # if (len(machine[relative_machine].items) == 1) | (last_job[relative_machine] == 0):
            #     ind_processing_job = 0
            #     setup_time.append(0)
            # else:
            # for job in machine[relative_machine].items:
            #     setuptime = setupTime[job.type - 1][int(last_job[relative_machine]) - 1]
            #     # priority_list.append(job.dueDate[job.numberOfOperations])
            #     job_queue_priority = choose_job_queue(weights_new, machine_number,
            #                                           job.processingTime[job.currentOperation - 1],
            #                                           job.dueDate[job.numberOfOperations], env, setuptime,
            #                                           job.priority)
            #     priority_list.append(job_queue_priority)
            #     setup_time.append(setuptime)
            # ind_processing_job = priority_list.index(max(priority_list))

            ind_processing_job = 0
            next_job = machine[relative_machine].items[ind_processing_job]
            setuptime = setupTime[next_job.type - 1][int(last_job[relative_machine]) - 1]
            tip = next_job.processingTime[next_job.currentOperation - 1] / 60 + setuptime
            makespan[relative_machine] = set_makespan(makespan[relative_machine], next_job, last_job[relative_machine],
                                                      env, setuptime)
            last_job[relative_machine] = next_job.type
            # schedule.append(
            #     [relative_machine, next_job.name, next_job.type, makespan[relative_machine], tip, setuptime, env.now,
            #      next_job.priority])
            machine[relative_machine].items.remove(next_job)
            yield env.timeout(tip)
            next_workstation(next_job, job_shop, env, all_store)
        else:
            yield job_shop.condition_flag[current_WC - 1][relative_machine]
            job_shop.condition_flag[current_WC - 1][relative_machine] = simpy.Event(env)


def cfp_wc(env, last_job, machine, makespan, store, job_shop, currentWC):
    while True:
        if store.items:
            c = bid_winner(env, store.items, machinesPerWC[currentWC], currentWC + 1, job_shop, last_job, makespan,
                           machine, store)
            env.process(c)
        tib = 0.1
        yield env.timeout(tib)


def no_in_system(R):
    """Total number of jobs in the resource R"""
    return len(R.put_queue) + len(R.users)


def source(env, number1, interval, job_shop, due_date_setting):
    if not noJobCap:  # If there is a limit on the number of jobs
        for ii in range(number1):
            job = New_Job('job%02d' % ii, env, ii, due_date_setting)
            firstWC = operationOrder[job.type - 1][0]
            store = eval('job_shop.storeWC' + str(firstWC))
            store.put(job)
            # d = job_pool_agent(job, firstWC, job_shop, store)
            # env.process(d)
            tib = random.expovariate(1.0 / interval)
            yield env.timeout(tib)
    else:
        while True:  # Needed for infinite case as True refers to "until".
            ii = number1
            number1 += 1
            job = New_Job('job%02d' % ii, env, ii, due_date_setting)
            job_shop.tardiness.append(-1)
            job_shop.makespan.append(0)
            job_shop.WIP += 1
            # print(job.type)
            firstWC = operationOrder[job.type - 1][0]
            store = eval('job_shop.storeWC' + str(firstWC))
            store.put(job)
            # d = job_pool_agent(job, firstWC, job_shop, store)
            # env.process(d)
            tib = random.uniform(0.67, 1)
            # tib = random.uniform(0.8, 1.2)
            yield env.timeout(tib)


class jobShop:
    def __init__(self, env, weights):
        machine_wc1 = {ii: Store(env) for ii in range(machinesPerWC[0])}
        machine_wc2 = {ii: Store(env) for ii in range(machinesPerWC[1])}
        machine_wc3 = {ii: Store(env) for ii in range(machinesPerWC[2])}
        machine_wc4 = {ii: Store(env) for ii in range(machinesPerWC[3])}
        # machine_wc5 = {ii: Store(env) for ii in range(machinesPerWC[4])}

        job_poolwc1 = simpy.FilterStore(env)
        job_poolwc2 = simpy.FilterStore(env)
        job_poolwc3 = simpy.FilterStore(env)
        job_poolwc4 = simpy.FilterStore(env)
        # job_poolwc5 = simpy.FilterStore(env)

        self.machinesWC1 = machine_wc1
        self.machinesWC2 = machine_wc2
        self.machinesWC3 = machine_wc3
        self.machinesWC4 = machine_wc4
        # self.machinesWC5 = machine_wc5

        self.QueuesWC1 = []
        self.QueuesWC2 = []
        self.QueuesWC3 = []
        self.QueuesWC4 = []
        # self.QueuesWC5 = []

        self.scheduleWC1 = []
        self.scheduleWC2 = []
        self.scheduleWC3 = []
        self.scheduleWC4 = []
        # self.scheduleWC5 = []

        self.condition_flag = []
        for wc in range(len(machinesPerWC)):
            Q = eval('self.QueuesWC' + str(wc + 1))
            self.condition_flag.append([])
            for ii in range(machinesPerWC[wc]):
                Q.append([])
                self.condition_flag[wc].append(simpy.Event(env))

        self.makespanWC1 = np.zeros(machinesPerWC[0])
        self.makespanWC2 = np.zeros(machinesPerWC[1])
        self.makespanWC3 = np.zeros(machinesPerWC[2])
        self.makespanWC4 = np.zeros(machinesPerWC[3])
        # self.makespanWC5 = np.zeros(machinesPerWC[4])

        self.last_job_WC1 = np.zeros(machinesPerWC[0])
        self.last_job_WC2 = np.zeros(machinesPerWC[1])
        self.last_job_WC3 = np.zeros(machinesPerWC[2])
        self.last_job_WC4 = np.zeros(machinesPerWC[3])
        # self.last_job_WC5 = np.zeros(machinesPerWC[4])

        self.storeWC1 = job_poolwc1
        self.storeWC2 = job_poolwc2
        self.storeWC3 = job_poolwc3
        self.storeWC4 = job_poolwc4
        # self.storeWC5 = job_poolwc5

        self.test_weights = weights
        self.makespan = []
        self.tardiness = []
        self.WIP = 0
        self.early_termination = 0

        self.bids = []
        self.winningbids = []


class New_Job:
    def __init__(self, name, env, number1, dueDateTightness):
        jobType = random.choices([1, 2, 3, 4], weights=demand, k=1)
        jobWeight = random.choices([1, 1, 1], weights=[0.5, 0.3, 0.2], k=1)
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
            self.dueDate[ii + 1] = self.dueDate[ii] + self.processingTime[ii]/60 * dueDateTightness
        # rushOrderProb = random.choices([0, 0.3], weights=[0.90, 0.10], k=1)
        # adjusted_Due_Date_Tightness = (self.dueDate[self.numberOfOperations] - np.nanmean(job_shop.makespan) * rushOrderProb[0] - env.now) / (self.dueDate[self.numberOfOperations] - env.now) * dueDateTightness
        # if adjusted_Due_Date_Tightness != dueDateTightness:
        #     for ii in range(self.numberOfOperations):
        #         meanPT = processingTimes[self.type - 1][ii]
        #         self.processingTime[ii] = meanPT
        #         # self.processingTime[ii] = random.gammavariate(5, meanPT / 5)
        #         self.dueDate[ii + 1] = self.dueDate[ii] + self.processingTime[ii] * adjusted_Due_Date_Tightness


def do_simulation_with_weights(mean_weight_new, std_weight_new, arrivalMean, due_date_tightness, iii):
    eta_new = np.zeros((sum(machinesPerWC), totalAttributes))
    objective_new = np.zeros(2)
    test_weights_pos = np.zeros((sum(machinesPerWC), totalAttributes))
    test_weights_min = np.zeros((sum(machinesPerWC), totalAttributes))
    for mm in range(sum(machinesPerWC)):
        for jj in range(totalAttributes):
            eta_new[mm][jj] = random.gauss(0, np.exp(std_weight_new[mm][jj]))
            test_weights_pos[mm][jj] = mean_weight_new[mm][jj] + (eta_new[mm][jj])
            test_weights_min[mm][jj] = mean_weight_new[mm][jj] - (eta_new[mm][jj])

    # seed = random.randrange(sys.maxsize)
    # random.seed(seed)
    # print("Seed was:", seed)
    env = Environment()
    job_shop = jobShop(env, test_weights_pos)
    env.process(source(env, number, arrivalMean, job_shop, due_date_tightness))
    all_stores = []

    for wc in range(len(machinesPerWC)):
        last_job = eval('job_shop.last_job_WC' + str(wc + 1))
        machine = eval('job_shop.machinesWC' + str(wc + 1))
        makespan = eval('job_shop.makespanWC' + str(wc + 1))
        store = eval('job_shop.storeWC' + str(wc + 1))
        all_stores.append(store)

        env.process(cfp_wc(env, last_job, machine, makespan, store, job_shop, wc))

        for ii in range(machinesPerWC[wc]):
            env.process(
                machine_processing(job_shop, wc + 1, machine_number_WC[wc][ii], env, test_weights_pos, ii, last_job,
                                   machine, makespan, all_stores))

    job_shop.end_event = env.event()

    env.run(until=job_shop.end_event)
    if job_shop.early_termination == 1:
        objective_new[0] = np.mean(np.nonzero(job_shop.tardiness[499:2999])) + 10_000 - np.count_nonzero(
            job_shop.makespan[499:2499])
    else:
        objective_new[0] = np.mean(job_shop.tardiness[499:2499])

    # seed = random.randrange(sys.maxsize)
    # random.seed(seed)
    # print("Seed was:", seed)
    env = Environment()
    job_shop = jobShop(env, test_weights_min)
    env.process(source(env, number, arrivalMean, job_shop, due_date_tightness))
    all_stores = []

    for wc in range(len(machinesPerWC)):
        last_job = eval('job_shop.last_job_WC' + str(wc + 1))
        machine = eval('job_shop.machinesWC' + str(wc + 1))
        makespan = eval('job_shop.makespanWC' + str(wc + 1))
        store = eval('job_shop.storeWC' + str(wc + 1))
        all_stores.append(store)

        env.process(cfp_wc(env, last_job, machine, makespan, store, job_shop, wc))

        for ii in range(machinesPerWC[wc]):
            env.process(
                machine_processing(job_shop, wc + 1, machine_number_WC[wc][ii], env, test_weights_min, ii, last_job,
                                   machine, makespan, all_stores))

    job_shop.end_event = env.event()

    env.run(until=job_shop.end_event)
    if job_shop.early_termination == 1:
        objective_new[1] = np.mean(np.nonzero(job_shop.tardiness[499:2499])) + 10_000 - np.count_nonzero(
            job_shop.makespan[499:2499])
    else:
        objective_new[1] = np.mean(job_shop.tardiness[499:2499])
    # print(len(job_shop.tardiness))

    return objective_new, eta_new


def run_linear(filename1, filename2, filename3, arrival_time_mean, due_date_k, alpha):
    file1 = open(filename1, "w")
    mean_weight = np.zeros((sum(machinesPerWC), totalAttributes))
    std_weight = np.zeros((sum(machinesPerWC), totalAttributes))
    for m in range(sum(machinesPerWC)):
        for j in range(totalAttributes):
            if j < 8:
                std_weight[m][j] = std_weight[m][j] + np.log(0.3)
            else:
                std_weight[m][j] = std_weight[m][j] + np.log(0.3)
    population_size = 12

    for i in range(sum(machinesPerWC)):
        mean_weight[i][0] = -0.5
        # mean_weight[i][7] = -1
        # mean_weight[i][2] = -3
        mean_weight[i][5] = -3
        mean_weight[i][6] = -3

        # mean_weight[i][noAttributes] = -1
        # mean_weight[i][noAttributes + 2] = -3

    jobshop_pool = Pool(processes=population_size)
    alpha_mean = 0.1
    alpha_std = 0.025
    beta_1 = 0.9
    beta_2 = 0.999
    m_t_mean = np.zeros((sum(machinesPerWC), totalAttributes))
    v_t_mean = np.zeros((sum(machinesPerWC), totalAttributes))

    m_t_std = np.zeros((sum(machinesPerWC), totalAttributes))
    v_t_std = np.zeros((sum(machinesPerWC), totalAttributes))

    objective = np.zeros((population_size, 2))
    eta = np.zeros((population_size, sum(machinesPerWC), totalAttributes))
    for num_sim in range(no_generation):
        for ii in range(1):
            seeds = range(int(population_size))
            func1 = partial(do_simulation_with_weights, mean_weight, std_weight, arrival_time_mean, due_date_k)
            makespan_per_seed = jobshop_pool.map(func1, seeds)
            for h, j in zip(range(int(population_size)), seeds):
                objective[ii * int(population_size) + j] = makespan_per_seed[h][0]
                eta[ii * int(population_size) + j] = makespan_per_seed[h][1]

        # print(objective)

        objective_norm = np.zeros((population_size, 2))
        # Normalise the current populations performance
        for ii in range(population_size):
            objective_norm[ii][0] = (objective[ii][0] - np.mean(objective, axis=0)[0]) / np.std(objective, axis=0)[0]

            objective_norm[ii][1] = (objective[ii][1] - np.mean(objective, axis=0)[1]) / np.std(objective, axis=0)[1]

        delta_mean_final = np.zeros((sum(machinesPerWC), totalAttributes))
        delta_std_final = np.zeros((sum(machinesPerWC), totalAttributes))
        for m in range(sum(machinesPerWC)):
            for j in range(totalAttributes):
                delta_mean = 0
                delta_std = 0
                for ii in range(population_size):
                    delta_mean += ((objective_norm[ii][0] - objective_norm[ii][1]) / 2) * eta[ii][m][j] / np.exp(
                        std_weight[m][j])

                    delta_std += ((objective_norm[ii][0] + objective_norm[ii][1]) / 2) * (eta[ii][m][j] ** 2 - np.exp(
                        std_weight[m][j])) / (np.exp(std_weight[m][j]))

                delta_mean_final[m][j] = delta_mean / population_size
                delta_std_final[m][j] = delta_std / population_size

        # print(delta_std_final[0:15][11])

        # print(delta_std_final)
        t = num_sim + 1
        for m in range(sum(machinesPerWC)):
            for j in range(totalAttributes):
                m_t_mean[m][j] = (beta_1 * m_t_mean[m][j] + (1 - beta_1) * delta_mean_final[m][j])
                v_t_mean[m][j] = (beta_2 * v_t_mean[m][j] + (1 - beta_2) * delta_mean_final[m][j] ** 2)
                m_hat_t = (m_t_mean[m][j] / (1 - beta_1 ** t))
                v_hat_t = (v_t_mean[m][j] / (1 - beta_2 ** t))
                mean_weight[m][j] = mean_weight[m][j] - (alpha_mean * m_hat_t) / (np.sqrt(v_hat_t) + 10 ** -8)

                # std_weight[m][j] = std_weight[m][j] - alpha_std * delta_std_final[m][j]
                #
                m_t_std[m][j] = (beta_1 * m_t_std[m][j] + (1 - beta_1) * delta_std_final[m][j])
                v_t_std[m][j] = (beta_2 * v_t_std[m][j] + (1 - beta_2) * delta_std_final[m][j] ** 2)
                m_hat_t_1 = (m_t_std[m][j] / (1 - beta_1 ** t))
                v_hat_t_1 = (v_t_std[m][j] / (1 - beta_2 ** t))
                std_weight[m][j] = std_weight[m][j] - (alpha_std * m_hat_t_1) / (np.sqrt(v_hat_t_1) + 10 ** -8)

        alpha_mean = 0.1 * np.exp(-(t - 1) / alpha)
        alpha_std = 0.025 * np.exp(-(t - 1) / alpha)

        #
        # final_objective.append(np.mean(np.mean(objective, axis=0)))
        # Ln.set_ydata(final_objective)
        # Ln.set_xdata(range(len(final_objective)))
        # plt.pause(1)

        print(num_sim, np.mean(np.mean(objective, axis=0)), np.mean(np.mean(np.exp(std_weight))))
        # print(np.mean(np.exp(std_weight), axis=0))
        L = [str(num_sim) + " ", str(np.mean(np.mean(objective, axis=0))) + " ",
             str(np.mean(np.exp(std_weight), axis=0)) + "\n"]
        file1.writelines(L)

        if num_sim == no_generation - 1:
            print(np.exp(std_weight))
            file1.close()
            file2 = open(filename2, 'w')
            writer = csv.writer(file2)
            writer.writerows(mean_weight)
            file2.close()

            file3 = open('filename3', 'w')
            writer = csv.writer(file3)
            writer.writerows(std_weight)
            file3.close()


if __name__ == '__main__':
    arrival_time = [0.30, 1.5429, 1.5429, 1.4572, 1.4572, 1.4572, 1.3804, 1.3804, 1.3804]
    utilization = [85, 85, 85, 90, 90, 90, 95, 95, 95]
    due_date_settings = [4, 6, 8, 4, 6, 8, 4, 6, 8]
    learning_decay_rate = [10, 100, 500, 1000, 2500, 5000, 10000]
    for num_runs in range(1):
        print("Current run is:" + str(utilization[num_runs]) + "-" + str(due_date_settings[num_runs]) + "-" + str(
            learning_decay_rate[5]))
        str1 = "Runs/Learning_Rate_Runs/Run-Custom10-" + str(utilization[num_runs]) + "-" + str(
            due_date_settings[num_runs]) + "-" + str(str(learning_decay_rate[4])) + ".txt"
        str2 = "Runs/Learning_Rate_Runs/Run-weights-Custom10-" + str(utilization[num_runs]) + "-" + str(
            due_date_settings[num_runs]) + "-" + str(str(learning_decay_rate[4])) + ".csv"
        str3 = "Runs/Learning_Rate_Runs/Run-std-Custom10-" + str(utilization[num_runs]) + "-" + str(
            due_date_settings[num_runs]) + "-" + str(str(learning_decay_rate[4])) + ".csv"
        run_linear(str1, str2, str3, arrival_time[num_runs], due_date_settings[num_runs], learning_decay_rate[5])

    # # to run GUI event loop
    # dat = [0, 1]
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # Ln, = ax.plot(dat)
    # ax.set_xlim([0, 1000])
    # ax.set_ylim([0, 150])
    # plt.ion()
    # plt.show()
    #
    # # setting title
    # plt.title("Mean objective function", fontsize=20)
    #
    # # setting x-axis label and y-axis label
    # plt.xlabel("No. of iterations")
    # plt.ylabel("Obejctive Function")