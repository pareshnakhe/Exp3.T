from prob import *
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import pylab
import threading
from operator import add

def make_bias_list(numRounds):
    bias = list()
    #Choose the biases of two actions from separate intervals

    bias.append(np.random.uniform(0.5, 0.6, numRounds) )
    bias.append(np.random.uniform(0.40, 0.5, numRounds))
    return bias

#Generates a list of biases with changes in trend as given by "break_pt_list"
def make_bias_list_switch(lm, input_bias, break_point_list):
    numActions = lm.get_numActions()

    #bias() is a list: [ [action 1 losses], [actn 2 losses] ...]
    bias = list()
    #Create a list of lists trend = [[], [] ...]
    #trend[i] holds biases for the entire time period for action i
    trend = list()
    for i in range(numActions):
        trend.append([])

    base = 0
    temp = 0
    for i in range(len(break_point_list)):
        for j in range(numActions):
            index = (base + j) % numActions
            trend[j].extend((np.random.uniform(input_bias[index][0], input_bias[index][1], break_point_list[i] - temp)).tolist())
        base = (base + 1) % numActions
        temp = break_point_list[i]

    for i in range(numActions):
        bias.append(trend[i])
return bias

#NOTE: Generating ARG losses is difficult since it is difficult to maintain the ARG property without making up a trivial example.
#For this reason, we pick loss vales from a truncated Guassian distribution and approximate the value of \Delta_{sp} by the
#gap in expected loss.
def generate_ARG_losses(lm, break_point_list, option):
    numActions = lm.get_numActions()

    lossVector = list()
    input_range = list()

    #Setting parameters for input range of each action
    if option == 2:
        mu_base = 0.450
        sigma = 0.05
        input_range.append([0.30, sigma])
    else:
        mu_base = 0.650
        sigma = 0.05
        input_range.append([0.30, sigma])

    for i in range(1, numActions):
            input_range.append([mu_base + i * 0.010, sigma])

    #**** Delta_{sp} is taken as (mu1 - mu2) / 2
    lm.Delta = (mu_base - input_range[0][0]) / 2.0
    #loss_bound does help us with faster trend detection
    #This is additional information available to us
    lm.loss_bound = 0.4
    print input_range
     ############################

    #Create a list of lists trend = [[], [] ...]
    #trend[i] holds loss values for the entire time period for action i
    trend = list()
    for i in range(numActions):
        trend.append([])

    base = 0
    temp = 0
    for i in range(len(break_point_list)):
        for j in range(numActions):
            index = (base + j) % numActions
            hold = (np.random.normal(input_range[index][0], input_range[index][1], break_point_list[i] - temp)).tolist()
            #Just ensuring that losses are in [0,1] range
            for itr in range(len(hold)):
                if hold[itr] < max(input_range[index][0] - 0.2, 0.0):
                    hold[itr] = max(input_range[index][0] - 0.2, 0.0)
                if hold[itr] > min(input_range[index][0] + 0.2, 1.0):
                    hold[itr] = min(input_range[index][0] + 0.2, 1.0)

            trend[j].extend(hold)
        base = (base + 1) % numActions
        temp = break_point_list[i]

    for i in range(numActions):
        lossVector.append(trend[i])

    return lossVector


#Generates actual loss vector on which (any) algorithm runs.
#Output: lossVector. lossVector[action][t] returns the loss of action at round t
def generate_stochastic_losses(lm, break_point_list, option):

    numRounds = lm.get_numRounds()
    numActions = lm.get_numActions()
    input_bias = list()

    if option == 0:
        mu_base = 0.450
        #The numbers in input_bias indicate the range of variation of bias for the particular action
        #input_bias.append([0.10, 0.15])
        input_bias.append([0.10, 0.1001])
    else:
        mu_base = 0.650
        #The numbers in input_bias indicate the range of variation of bias for the particular action
        #input_bias.append([0.10, 0.15])
        input_bias.append([0.10, 0.1001])

    for i in range(1, numActions):
        #input_bias.append([mu_base, np.random.uniform(mu_base, mu_base + 0.10)])
        input_bias.append([mu_base, np.random.uniform(mu_base, mu_base + 0.1001)])

    #Changed from 4.0 to 2.0
    lm.Delta = (mu_base - input_bias[0][0]) / 4.0
    print input_bias

    if break_point_list:
        biasList = make_bias_list_switch(lm, input_bias, break_point_list)
    else:
        biasList = make_bias_list(numRounds)
#==============================
    #print biasList
    lossVector = list()
    for bList in biasList:
        lossVector.append([1 if random.random() < bias else 0 for bias in bList])

    return lossVector


def exp3S_engine(lm, accessLoss):
    numActions = lm.get_numActions()
    numRounds = lm.get_numRounds()


    weights = [1.0] * numActions
    gamma = math.sqrt(numActions * math.log(numActions * numRounds) / float(numRounds))
    alpha = 1.0 / numRounds
    t = 1
    while True:
        if t % 200000 == 0:
            #print "adjusted"
            weights = [wt / sum(weights) for wt in weights]

        forecaster = exp3S_distr(weights, gamma)
        chosenAction = Bubeck_draw(forecaster, t)
        if chosenAction is None:
            #print sum(forecaster)
            #print t, chosenAction
            exit(-1)
        crnt_loss = accessLoss(chosenAction, t)
        estimatedLoss = (float(1 - crnt_loss)) / forecaster[chosenAction]

        sum_wts = sum(weights)
        weights[chosenAction] *= math.exp((estimatedLoss * gamma) / numActions)
        for action in range(len(weights)):
            weights[action] += ((math.e * alpha) / float(numActions)) * sum_wts
        t += 1
        yield crnt_loss, weights, []


def old_exp3_engine(lm, accessLoss):
    numActions = lm.get_numActions()
    numRounds = lm.get_numRounds()

    weights = [1.0] * numActions
    gamma = math.sqrt((numActions * math.log(numActions)) / ((math.e - 1) * numRounds))

    t = 1
    while True:
        #print weights
        #If we run this algorithm for long enough, some of the weights are becoming so large that
        #they are being represented by "inf". Run breaks at this point.
        forecaster = distr(weights, gamma, t)
        chosenAction = draw(forecaster)
        crnt_loss = accessLoss(chosenAction, t)
        estimatedLoss = (float(1 - crnt_loss)) / forecaster[chosenAction]
        weights[chosenAction] *= math.exp((estimatedLoss * gamma) / numActions)
        t += 1
        yield crnt_loss, weights, []



#Runs Exp3 on loss model where the best action switches according to break_point_list
def Exp3Switch(lossVector, lm, break_point_list, engine):

    numActions = lm.get_numActions()
    numRounds = lm.get_numRounds()
    accessLoss = lambda action, round: lossVector[action][round]

    #break_point_list holds a list of rounds where trend changes
    #optimal_action_switch_list holds a list of rounds where the action of benchmark strategy switches
    #In normal case, we'll keep both the lists same
    #break_point_list = [20000, numRounds]
    optimal_action_switch_list = break_point_list
    bestAction = list()

    #lossVector = generate_stochastic_losses(numRounds, break_point_list)
    #print lossVector
    #accessLoss = lambda action, round: lossVector[action][round]

    start_rnd = 1
    for change_pt in optimal_action_switch_list:
        min_loss_action = min(range(numActions), key=lambda action: sum([lossVector[action][rnd] for rnd in range(start_rnd, change_pt)]))
        bestAction.append(min_loss_action)
        start_rnd = change_pt + 1

    cumulativeLoss = 0
    bestActionCumulativeLoss = 0
    weakRegret = []

    t = 1
    index = 0
    hold = optimal_action_switch_list[index]
    #print hold, optimal_action_switch_list
    #for (crnt_loss, weights, eta_list) in Bubeck_exp3():
    #for (crnt_loss, weights, eta_list) in exp3S_engine():
    for (crnt_loss, weights, eta_list) in engine(lm, accessLoss):

        if t >= numRounds-1:
            break

        if t > hold:
            index += 1
            print t, index
            hold = optimal_action_switch_list[index]

        crnt_bestAction = bestAction[index]

        cumulativeLoss += crnt_loss
        bestActionCumulativeLoss += lossVector[crnt_bestAction][t]

        weakRegret.append(cumulativeLoss - bestActionCumulativeLoss)
        #regretBound = (numActions / 2) * sum(eta_list) + math.log(numActions) / eta_list[t]

        #print("regret: %d\tmaxRegret: %.2f\tweights: (%s)" % (weakRegret[-1], regretBound, ', '.join(["%.3f" % weight for weight in distr(weights)])))
        t += 1

    #print cumulativeLoss, bestActionCumulativeLoss
    #plotGraph(weakRegret, numActions)
    return weakRegret


def plotGraph(base, S, D):

    pylab.plot(base, '-b', label='Base Exp3')
    pylab.plot(S, '-r', label='Exp3S')
    pylab.plot(D, '-g', label='Exp3D')
    pylab.legend(loc='upper left')
    pylab.show()

    #plt.plot(regret1, 'r--', regret2, 'b--')
    #plt.show()

class loss_monitor:

    def __init__(self, numActions, numRounds):

        #primary data structures used
        self.history = [[], [], []]
        self.crnt_interval_loss = [0] * numActions
        self.crnt_interval_index = 0

        #Game parameters
        self.K = numActions
        self.T = numRounds
        self.kt_star = 0
        self.reset_hold = 0
        self.Delta = 0
        self.loss_bound = 1
        self.global_reset = 0

    def clean_object(self):
        self.history = [[], [], []]
        self.crnt_interval_loss = [0] * self.K
        self.crnt_interval_index = 0
        self.reset_hold = 0
        self.global_reset = 0

    def get_numActions(self):
        return self.K

    def get_numRounds(self):
        return self.T

    def get_global_reset(self):
        return self.global_reset

    def set_global_reset(self, val):
        self.global_reset = val

    def get_kt_star(self):
        delta = math.sqrt((self.K / (self.T * math.log(self.K))))
        A = math.log(2.0 * self.K / delta)
        #B = 2 * math.pow(self.Delta, 2)
        B = min(((8.0 * math.pow(self.Delta, 2)) / math.pow(self.loss_bound, 2.0)), 1)
        return int(self.K * (A / B))

    def get_gamma(self):
        return math.sqrt((self.kt_star * math.log(self.K) / self.T))

    def interval_length(self):
        # |I| = K t^* / \gamma
        self.kt_star = self.get_kt_star()
        gamma = self.get_gamma()
        return int(self.kt_star / gamma)

    def make_schedule(self):
        schedule = list()
        lm_plays_set = set()
        num_lm_plays = self.get_kt_star()
        inter_len = self.interval_length()
        print "interval length = %d" % inter_len

        for interval in range(int(np.floor(self.T / inter_len))):
            lm_plays = random.sample(range(inter_len), num_lm_plays)
            lm_plays_set = set(lm_plays)
            action = 0
            for i in range(inter_len):
                if i in lm_plays_set:
                    lm_plays.remove(i)
                    schedule.append(action)
                    action = (action + 1) % self.K
                else:
                    schedule.append(-1)

        #The last incomplete interval is filled with all Exp3 plays for each round.
        for i in range(len(schedule), self.T):
            schedule.append(-1)
        return schedule

    def monitor_loss(self, action, loss):
        self.crnt_interval_loss[action] += loss

    def reset_exp3(self):
        print "Reset"
        #global_reset is a flag for Bubeck_exp3_mod to restart
        self.set_global_reset(1)
        self.reset_hold = 1

    def trend_detection(self):
        #'flag' is an indicator variable to check if exp3 was reseted in this call
        flag = 0
        t_star = int(self.get_kt_star() / self.K)
        self.history[self.crnt_interval_index % 3] = [float(loss) / t_star for loss in self.crnt_interval_loss]

        if self.reset_hold == 1:
            print "trend detection skipped"
            self.reset_hold = 0
        else:
            #print "trend detection running"

            if self.history[0] and self.history[1] and self.history[2]:
                min_loss_action_crnt = min(range(self.K), key=lambda action: self.history[self.crnt_interval_index % 3][action])
                min_loss_action_old = min(range(self.K), key=lambda action: self.history[(self.crnt_interval_index + 1) % 3][action])

                if min_loss_action_crnt != min_loss_action_old:
                    self.reset_exp3()
                    flag = 1

        self.crnt_interval_index += 1
        self.crnt_interval_loss = [0] * self.K
        return flag


#This functions runs either
def Bubeck_exp3_mod(lm, loss, schedule):
    numActions = lm.get_numActions()

    weights = [1.0] * numActions
    #eta_t = lambda time: math.sqrt(math.log(numActions) / (time * numActions))
    #eta = math.sqrt(math.log(numActions) / (numRounds * numActions))
    t = 1
    exp3_cntr = 1
    eta_list = [0.0]

    while True:
        if lm.get_global_reset() == 1:
            #Should we also reset 't'
            weights = [1.0] * numActions
            lm.set_global_reset(0)
            exp3_cntr = 1

        # Run Exp3
        if schedule[t] == -1:
            #eta_list.append(eta_t(t))
            forecaster = Bubeck_distr(weights)
            chosenAction = Bubeck_draw(forecaster)
            crnt_loss = loss(chosenAction, t)

            eta = math.sqrt(math.log(numActions) / (exp3_cntr * numActions))

            estimatedLoss = (1.0 * crnt_loss) / forecaster[chosenAction]
            #weights[chosenAction] *= math.exp(-1.0 * estimatedLoss * eta_t(t)) # important that we use estimated reward here!
            weights[chosenAction] *= math.exp(-1.0 * estimatedLoss * eta)
        else:
            #Run Loss monitoring component
            lm_action = schedule[t]
            crnt_loss = loss(lm_action, t)
            lm.monitor_loss(lm_action, crnt_loss)

        exp3_cntr += 1
        t += 1
        yield crnt_loss, weights, eta_list



#Runs Exp3 + loss_monitoring on loss model where the best action switches according to break_point_list
def Exp3D(lossVector, lm, break_point_list):
    numActions = lm.get_numActions()
    numRounds = lm.get_numRounds()
    accessLoss = lambda action, round: lossVector[action][round]

    #optimal_action_switch_list holds a list of rounds where the action of benchmark strategy switches
    #In normal case, we'll keep both the lists same
    optimal_action_switch_list = break_point_list
    bestAction = list()

    #Here we are computing the benchmark strategy
    start_rnd = 1
    for change_pt in optimal_action_switch_list:
        min_loss_action = min(range(numActions), key=lambda action: sum([lossVector[action][rnd] for rnd in range(start_rnd, change_pt)]))
        bestAction.append(min_loss_action)
        start_rnd = change_pt + 1

    print bestAction
    cumulativeLoss = 0
    bestActionCumulativeLoss = 0
    weakRegret = []

    t = 1
    index = 0
    int_cntr = 0
    hold = optimal_action_switch_list[index]
    #print hold, optimal_action_switch_list
    schedule = lm.make_schedule()
    for (crnt_loss, weights, eta_list) in Bubeck_exp3_mod(lm, accessLoss, schedule):

        if t >= numRounds - 1:
            break

        if t > hold:
            index += 1
            print t, index
            hold = optimal_action_switch_list[index]

        crnt_bestAction = bestAction[index]

        cumulativeLoss += crnt_loss
        bestActionCumulativeLoss += lossVector[crnt_bestAction][t]

        weakRegret.append(cumulativeLoss - bestActionCumulativeLoss)
        #regretBound = (numActions / 2) * sum(eta_list) + math.log(numActions) / eta_list[t]

        #print("regret: %d\tweights: (%s)" % (weakRegret[-1], ', '.join(["%.3f" % weight for weight in distr(weights)])))
        t += 1
        int_cntr += 1
        if int_cntr == lm.interval_length():
            #print "Trend detection called: %d" % t
            if lm.trend_detection():
                print "Resetted at round: %d" % t
            int_cntr = 0


    #print cumulativeLoss, bestActionCumulativeLoss
    return weakRegret


def sw_ucb_engine(lm, lossVector):
    numActions = lm.get_numActions()
    numRounds = lm.get_numRounds()
    accessLoss = lambda action, round: lossVector[action][round]

    #parameter of the algorithm
    tau = int(2.0 * math.sqrt(numRounds * math.log(numRounds)))
    print "Tau: %d" % tau

    # These lists keep track of the following:
    # For each round among the last tau rounds, which action was chosen and what was the loss incurred
    cnt_list = [[] for i in range(numActions)]
    loss_list = [[] for i in range(numActions)]
    ucb_list = [0.0] * numActions

    for i in xrange(numActions):
        cnt_list[i] = [0] * tau
        loss_list[i] = [0.0] * tau

    #Does t=0 create a problem??
    t = 0
    #Important
    #loss_list holds 1 loss was 0 and 0 when loss incurred was 1
    while True:
        chosenAction = ucb_list.index(max(ucb_list))
        crnt_loss = accessLoss(chosenAction, t)
        for k in xrange(numActions):
            if t < tau:
                if k == chosenAction:
                    cnt_list[k][t] = 1
                    loss_list[k][t] = 1 - crnt_loss
                    ucb_list[k] = float(sum(loss_list[k])) / (sum(cnt_list[k]) + 1) + math.sqrt(math.log(t+1) / (sum(cnt_list[k]) + 1))
                else:
                    cnt_list[k][t] = 0
                    loss_list[k][t] = 0
                    ucb_list[k] = float(sum(loss_list[k])) / (sum(cnt_list[k]) + 1) + math.sqrt(math.log(t+1) / (sum(cnt_list[k]) + 1))
            else:
                if k == chosenAction:
                    cnt_list[k][t % tau] = 1
                    loss_list[k][t % tau] = 1 - crnt_loss
                    ucb_list[k] = float(sum(loss_list[k])) / (sum(cnt_list[k]) + 1) + math.sqrt(math.log(tau) / (sum(cnt_list[k]) + 1))
                else:
                    cnt_list[k][t % tau] = 0
                    loss_list[k][t % tau] = 0
                    ucb_list[k] = float(sum(loss_list[k])) / (sum(cnt_list[k]) + 1) + math.sqrt(math.log(tau) / (sum(cnt_list[k]) + 1))

        t += 1
        yield crnt_loss

# This engine separates the exploration component from the forecaster.
# It also runs drift detection procedure within the main loop
def old_exp3_mod_engine(lm, accessLoss, option):
    numActions = lm.get_numActions()
    numRounds = lm.get_numRounds()
    weights = [1.0] * numActions

    #Parameters of the algorithm
    gamma = math.sqrt((numActions * math.log(numActions) * math.log(numRounds)) / numRounds)
    if option == 1:
        H = 35.0 * math.sqrt(numRounds * math.log(numRounds))
    else:
        H = 45.0 * math.sqrt(numRounds * math.log(numRounds))
    #Set epsilon according to algo
    delta = math.sqrt(math.log(numRounds) / float(numActions * numRounds))
    epsilon = math.sqrt((numActions * math.log(math.pow(delta, -1))) / (2.0 * gamma * H))

    #pull_cnt keeps track of the number of gamma-pulls of each action
    #emp_mean tracks the empirical mean of gamma samples
    pull_cnt = list()
    emp_mean = list()
    for i in xrange(numActions):
        pull_cnt.append(0)
        emp_mean.append(0.0)

    t = 1
    print "gamma: %f, H: %f, delta: %f, gamma-H-K: %f, epsilon: %f" % (gamma, H, delta, (gamma * H) / numActions, epsilon)
    while True:

        forecaster = distr(weights, gamma, t)
        rand_num = random.uniform(0, 1)
        if rand_num <= gamma:
            #Choose an action uniformly at random
            chosenAction = random.randint(0, numActions - 1)
            crnt_loss = accessLoss(chosenAction, t)
            #Update pull_cnt and emp_mean lists.
            temp = (emp_mean[chosenAction] * pull_cnt[chosenAction]) + crnt_loss
            pull_cnt[chosenAction] += 1
            emp_mean[chosenAction] = temp / pull_cnt[chosenAction]
        else:
            #IMP:forecaster distr here is the old one but for drawing we use "Bubeck_draw". This is bcoz of the separation of gamma component.
            chosenAction = Bubeck_draw(Bubeck_distr(weights))

        crnt_loss = accessLoss(chosenAction, t)
        estimatedLoss = (float(1 - crnt_loss)) / forecaster[chosenAction]
        weights[chosenAction] *= math.exp((estimatedLoss * gamma) / numActions)
        t += 1

        #Check for detection
        if min(pull_cnt) > (gamma * H) / float(numActions):
            #print emp_mean, pull_cnt
            k_star = forecaster.index(max(forecaster))
            for i in xrange(numActions):
                if i == k_star:
                    continue
                if emp_mean[i] - emp_mean[k_star] >= 2.0 * epsilon:
                    weights = [1.0] * numActions
                    print "Drift detected"


            pull_cnt = list()
            emp_mean = list()

            for i in xrange(numActions):
                pull_cnt.append(0)
                emp_mean.append(0.0)

        yield crnt_loss, weights



def SW_UCB(lossVector, lm, break_point_list):
    numActions = lm.get_numActions()
    numRounds = lm.get_numRounds()

    #optimal_action_switch_list holds a list of rounds where the action of benchmark strategy switches
    #In normal case, we'll keep both the lists same
    optimal_action_switch_list = break_point_list
    bestAction = list()

    #Here we are computing the benchmark strategy
    start_rnd = 1
    for change_pt in optimal_action_switch_list:
        min_loss_action = min(range(numActions), key=lambda action: sum([lossVector[action][rnd] for rnd in range(start_rnd, change_pt)]))
        bestAction.append(min_loss_action)
        start_rnd = change_pt + 1

    print bestAction
    cumulativeLoss = 0
    bestActionCumulativeLoss = 0
    weakRegret = []

    t = 0
    crnt_int_index = 0
    int_cntr = 0
    hold = optimal_action_switch_list[crnt_int_index]
    #print hold, optimal_action_switch_list

    for (crnt_loss) in sw_ucb_engine(lm, lossVector):

        if t >= numRounds - 1:
            break

        if t > hold:
            crnt_int_index += 1
            print t, crnt_int_index
            hold = optimal_action_switch_list[crnt_int_index]

        crnt_bestAction = bestAction[crnt_int_index]

        cumulativeLoss += crnt_loss
        bestActionCumulativeLoss += lossVector[crnt_bestAction][t]

        weakRegret.append(cumulativeLoss - bestActionCumulativeLoss)
        t += 1

    return weakRegret


def Exp3R(lossVector, lm, break_point_list, option):
    numActions = lm.get_numActions()
    numRounds = lm.get_numRounds()
    accessLoss = lambda action, round: lossVector[action][round]

    #optimal_action_switch_list holds a list of rounds where the action of benchmark strategy switches
    #In normal case, we'll keep both the lists same
    optimal_action_switch_list = break_point_list
    bestAction = list()

    #Here we are computing the benchmark strategy
    start_rnd = 1
    for change_pt in optimal_action_switch_list:
        min_loss_action = min(range(numActions), key=lambda action: sum([lossVector[action][rnd] for rnd in range(start_rnd, change_pt)]))
        bestAction.append(min_loss_action)
        start_rnd = change_pt + 1

    print bestAction
    cumulativeLoss = 0
    bestActionCumulativeLoss = 0
    weakRegret = []

    t = 1
    crnt_int_index = 0
    int_cntr = 0
    hold = optimal_action_switch_list[crnt_int_index]
    #print hold, optimal_action_switch_list

    for (crnt_loss, weights) in old_exp3_mod_engine(lm, accessLoss, option):

        if t >= numRounds - 1:
            break

        if t > hold:
            crnt_int_index += 1
            print t, crnt_int_index
            hold = optimal_action_switch_list[crnt_int_index]

        crnt_bestAction = bestAction[crnt_int_index]

        cumulativeLoss += crnt_loss
        bestActionCumulativeLoss += lossVector[crnt_bestAction][t]

        weakRegret.append(cumulativeLoss - bestActionCumulativeLoss)
        t += 1

    return weakRegret


def module(option, action_set):

    for k in action_set:
        print "With %d actions:" % k
        lm = loss_monitor(k, 100000)

        #break_point_list holds a list of rounds where trend changes
        break_point_list = [20000, 42000, lm.get_numRounds()]

        if option < 2:
            lossVector = generate_stochastic_losses(lm, break_point_list, option)
        else:
            lossVector = generate_ARG_losses(lm, break_point_list, option)

        numIterations = 1

        for itr in range(numIterations):
            print "Iteration number: %d" % itr
            regret_swucb = SW_UCB(lossVector, lm, break_point_list)
            lm.clean_object()
            regret_exp3r = Exp3R(lossVector, lm, break_point_list, option)
            lm.clean_object()
            regret_exp3base = Exp3Switch(lossVector, lm, break_point_list, old_exp3_engine)
            lm.clean_object()
            regret_exp3s = Exp3Switch(lossVector, lm, break_point_list, exp3S_engine)
            lm.clean_object()
            regret_exp3d = Exp3D(lossVector, lm, break_point_list)
            lm.clean_object()

        threadLock.acquire()
        pylab.xlabel("rounds")
        pylab.ylabel("Cumulative Regret")
        pylab.ticklabel_format(style='sci', scilimits=(0,0), axis='both')

        pylab.plot(regret_swucb, 'm', label='SW-UCB')
        pylab.plot(regret_exp3r, 'c', label='Exp3.R')
        pylab.plot(regret_exp3base, 'b', linewidth=1.0, label='Base Exp3')
        # #pylab.plot([regret_exp3base[i] + std_dev_exp3base[i] for i in range(len(regret_exp3base))], 'b-.')
        pylab.plot(regret_exp3s, 'r', label='Exp3.S')
        # #pylab.plot([regret_exp3s[i] + std_dev_exp3s[i] for i in range(len(regret_exp3s))], 'r-.')
        pylab.plot(regret_exp3d, 'g', linewidth=1.0, label='Exp3D')
        # #pylab.plot([regret_exp3d[i] + std_dev_exp3d[i] for i in range(len(regret_exp3d))], 'g-.')
        pylab.legend(loc='upper left')

        pylab.savefig('TEST: HOption=' + str(option) + ': Fig-' + str(k) + '-actions.png')
        #pylab.show()
        pylab.clf()
        threadLock.release()


class myThread (threading.Thread):
    def __init__(self, option, action_set):
        threading.Thread.__init__(self)
        self.option = option
        self.action_set = action_set
    def run(self):
        print "Starting " + self.name
        module(self.option, self.action_set)
        print "Exiting " + self.name


if __name__ == "__main__":
    action_set = [2]
    threadLock = threading.Lock()
    #Anonymous function to access loss of a given action in a given round
    #accessLoss = lambda action, round: lossVector[action][round]

    thread0 = myThread(0, action_set)
    thread1 = myThread(1, action_set)
    thread2 = myThread(2, action_set)
    thread3 = myThread(3, action_set)

    thread0.start()
    thread1.start()
    thread2.start()
    thread3.start()

    quit(1)