import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.prob import *


def exp3S_engine(lm, accessLoss):
    num_actions = lm.get_num_actions()
    num_rounds = lm.get_num_rounds()

    weights = [1.0] * num_actions
    gamma = math.sqrt(num_actions * math.log(num_actions * num_rounds) / float(num_rounds))
    alpha = 1.0 / num_rounds
    t = 1
    while True:
        if t % 200000 == 0:
            weights = [wt / sum(weights) for wt in weights]

        forecaster = exp3S_distr(weights, gamma)
        chosen_action = Bubeck_draw(forecaster, t)
        if chosen_action is None:
            print(t, chosen_action)
            exit(-1)
        crnt_loss = accessLoss(chosen_action, t)
        estimatedLoss = (float(1 - crnt_loss)) / forecaster[chosen_action]

        sum_wts = sum(weights)
        weights[chosen_action] *= math.exp((estimatedLoss * gamma) / num_actions)
        for action in range(len(weights)):
            weights[action] += ((math.e * alpha) / float(num_actions)) * sum_wts
        t += 1
        yield crnt_loss, weights, []


def old_exp3_engine(lm, access_loss):
    num_actions = lm.get_num_actions()
    num_rounds = lm.get_num_rounds()

    weights = [1.0] * num_actions
    gamma = math.sqrt((num_actions * math.log(num_actions)) / ((math.e - 1) * num_rounds))

    t = 1
    while True:
        # If we run this algorithm for long enough, weights might too large causing overflow.
        # Run breaks at this point.
        forecaster = distr(weights, gamma, t)
        chosen_action = draw(forecaster)
        crnt_loss = access_loss(chosen_action, t)
        estimated_loss = (float(1 - crnt_loss)) / forecaster[chosen_action]
        weights[chosen_action] *= math.exp((estimated_loss * gamma) / num_actions)
        t += 1
        yield crnt_loss, weights, []


def Exp3Switch(lossVector, lm, break_point_list, engine):
    """
    Runs Exp3 on loss model where the best action switches according to break_point_list
    """

    num_actions = lm.get_num_actions()
    num_rounds = lm.get_num_rounds()
    access_loss = lambda action, rnd: lossVector[action][rnd]

    # optimal_action_switch_list holds a list of rounds where the action of benchmark strategy switches
    optimal_action_switch_list = break_point_list
    best_action = list()

    start_rnd = 1
    for change_pt in optimal_action_switch_list:
        min_loss_action = min(
            range(num_actions),
            key=lambda action: sum([lossVector[action][rnd] for rnd in range(start_rnd, change_pt)])
        )
        best_action.append(min_loss_action)
        start_rnd = change_pt + 1

    cumulative_loss = 0
    best_action_cumulative_loss = 0
    weak_regret = []

    t = 1
    index = 0
    hold = optimal_action_switch_list[index]

    for (crnt_loss, weights, eta_list) in engine(lm, access_loss):

        if t >= num_rounds - 1:
            break

        if t > hold:
            index += 1
            print(t, index)
            hold = optimal_action_switch_list[index]

        crnt_best_action = best_action[index]

        cumulative_loss += crnt_loss
        best_action_cumulative_loss += lossVector[crnt_best_action][t]

        weak_regret.append(cumulative_loss - best_action_cumulative_loss)
        t += 1

    return weak_regret


def plot_graph(base, S, D):
    plt.plot(base, '-b', label='Base Exp3')
    plt.plot(S, '-r', label='Exp3S')
    plt.plot(D, '-g', label='Exp3D')
    plt.legend(loc='upper left')
    plt.show()


def Bubeck_exp3_mod(lm, loss, schedule):
    num_actions = lm.get_num_actions()

    weights = [1.0] * num_actions
    # eta_t = lambda time: math.sqrt(math.log(num_actions) / (time * num_actions))
    # eta = math.sqrt(math.log(num_actions) / (num_rounds * num_actions))
    t = 1
    exp3_cntr = 1
    eta_list = [0.0]

    while True:
        if lm.get_global_reset() == 1:
            weights = [1.0] * num_actions
            lm.set_global_reset(0)
            exp3_cntr = 1

        # Run Exp3
        if schedule[t] == -1:
            forecaster = Bubeck_distr(weights)
            chosen_action = Bubeck_draw(forecaster)
            crnt_loss = loss(chosen_action, t)

            eta = math.sqrt(math.log(num_actions) / (exp3_cntr * num_actions))
            estimated_loss = (1.0 * crnt_loss) / forecaster[chosen_action]
            weights[chosen_action] *= math.exp(-1.0 * estimated_loss * eta)
        else:
            # Run Loss monitoring component
            lm_action = schedule[t]
            crnt_loss = loss(lm_action, t)
            lm.monitor_loss(lm_action, crnt_loss)

        exp3_cntr += 1
        t += 1
        yield crnt_loss, weights, eta_list


def Exp3D(loss_vector, lm, break_point_list):
    """
    Exp3 + loss_monitoring on loss model where the best action
    switches according to break_point_list
    """
    num_actions = lm.get_num_actions()
    num_rounds = lm.get_num_rounds()
    access_loss = lambda action, rnd: loss_vector[action][rnd]

    # optimal_action_switch_list holds a list of rounds
    # where the action of benchmark strategy switches
    optimal_action_switch_list = break_point_list
    best_action = list()

    # Compute the benchmark strategy
    start_rnd = 1
    for change_pt in optimal_action_switch_list:
        min_loss_action = min(
            range(num_actions),
            key=lambda action: sum([loss_vector[action][rnd] for rnd in range(start_rnd, change_pt)])
        )
        best_action.append(min_loss_action)
        start_rnd = change_pt + 1

    print(best_action)
    cumulative_loss = 0
    best_action_cumulative_loss = 0
    weak_regret = []

    t = 1
    index = 0
    int_cntr = 0
    hold = optimal_action_switch_list[index]

    schedule = lm.make_schedule()
    for (crnt_loss, weights, eta_list) in Bubeck_exp3_mod(lm, access_loss, schedule):

        if t >= num_rounds - 1:
            break

        if t > hold:
            index += 1
            print(t, index)
            hold = optimal_action_switch_list[index]

        crnt_best_action = best_action[index]

        cumulative_loss += crnt_loss
        best_action_cumulative_loss += loss_vector[crnt_best_action][t]
        weak_regret.append(cumulative_loss - best_action_cumulative_loss)

        t += 1
        int_cntr += 1
        if int_cntr == lm.interval_length():
            if lm.trend_detection():
                print("Resetted at round: {}".format(t))
            int_cntr = 0

    return weak_regret


def sw_ucb_engine(lm, loss_vector):
    num_actions = lm.get_num_actions()
    num_rounds = lm.get_num_rounds()
    access_loss = lambda action, round: loss_vector[action][round]

    # parameter of the algorithm
    tau = int(2.0 * math.sqrt(num_rounds * math.log(num_rounds)))
    print("Tau: {}".format(tau))

    # These lists keep track of the following:
    # For each round among the last tau rounds, which action was chosen and what was the loss incurred
    cnt_list = [[] for i in range(num_actions)]
    loss_list = [[] for i in range(num_actions)]
    ucb_list = [0.0] * num_actions

    for i in range(num_actions):
        cnt_list[i] = [0] * tau
        loss_list[i] = [0.0] * tau

    t = 0
    # Important
    # loss_list holds 1 loss was 0 and 0 when loss incurred was 1
    while True:
        chosen_action = ucb_list.index(max(ucb_list))
        crnt_loss = access_loss(chosen_action, t)
        for k in range(num_actions):
            if t < tau:
                if k == chosen_action:
                    cnt_list[k][t] = 1
                    loss_list[k][t] = 1 - crnt_loss
                    ucb_list[k] = float(sum(loss_list[k])) / (sum(cnt_list[k]) + 1) + math.sqrt(
                        math.log(t + 1) / (sum(cnt_list[k]) + 1))
                else:
                    cnt_list[k][t] = 0
                    loss_list[k][t] = 0
                    ucb_list[k] = float(sum(loss_list[k])) / (sum(cnt_list[k]) + 1) + math.sqrt(
                        math.log(t + 1) / (sum(cnt_list[k]) + 1))
            else:
                if k == chosen_action:
                    cnt_list[k][t % tau] = 1
                    loss_list[k][t % tau] = 1 - crnt_loss
                    ucb_list[k] = float(sum(loss_list[k])) / (sum(cnt_list[k]) + 1) + math.sqrt(
                        math.log(tau) / (sum(cnt_list[k]) + 1))
                else:
                    cnt_list[k][t % tau] = 0
                    loss_list[k][t % tau] = 0
                    ucb_list[k] = float(sum(loss_list[k])) / (sum(cnt_list[k]) + 1) + math.sqrt(
                        math.log(tau) / (sum(cnt_list[k]) + 1))

        t += 1
        yield crnt_loss


def old_exp3_mod_engine(lm, access_loss, option):
    """
    This engine separates the exploration component from the forecaster.
    It also runs drift detection procedure within the main loop
    """
    num_actions = lm.get_num_actions()
    num_rounds = lm.get_num_rounds()
    weights = [1.0] * num_actions

    # Parameters of the algorithm
    gamma = math.sqrt((num_actions * math.log(num_actions) * math.log(num_rounds)) / num_rounds)
    if option == 1:
        H = 35.0 * math.sqrt(num_rounds * math.log(num_rounds))
    else:
        H = 45.0 * math.sqrt(num_rounds * math.log(num_rounds))
    # Set epsilon according to algo
    delta = math.sqrt(math.log(num_rounds) / float(num_actions * num_rounds))
    epsilon = math.sqrt((num_actions * math.log(math.pow(delta, -1))) / (2.0 * gamma * H))

    # pull_cnt keeps track of the number of gamma-pulls of each action
    # emp_mean tracks the empirical mean of gamma samples
    pull_cnt = list()
    emp_mean = list()
    for i in range(num_actions):
        pull_cnt.append(0)
        emp_mean.append(0.0)

    t = 1
    print("gamma: {}, H: {}, delta: {}, gamma-H-K: {}, epsilon: {}".format(gamma, H, delta, (gamma * H) / num_actions,
                                                                           epsilon))
    while True:
        forecaster = distr(weights, gamma, t)
        rand_num = random.uniform(0, 1)
        if rand_num <= gamma:
            # Choose an action uniformly at random
            chosen_action = random.randint(0, num_actions - 1)
            crnt_loss = access_loss(chosen_action, t)
            # Update pull_cnt and emp_mean lists.
            temp = (emp_mean[chosen_action] * pull_cnt[chosen_action]) + crnt_loss
            pull_cnt[chosen_action] += 1
            emp_mean[chosen_action] = temp / pull_cnt[chosen_action]
        else:
            # IMP:forecaster distr here is the old one but for drawing we use "Bubeck_draw".
			# This is bcoz of the separation of gamma component.
            chosen_action = Bubeck_draw(Bubeck_distr(weights))

        crnt_loss = access_loss(chosen_action, t)
        estimated_loss = (float(1 - crnt_loss)) / forecaster[chosen_action]
        weights[chosen_action] *= math.exp((estimated_loss * gamma) / num_actions)
        t += 1

        # Check for detection
        if min(pull_cnt) > (gamma * H) / float(num_actions):

            k_star = forecaster.index(max(forecaster))
            for i in range(num_actions):
                if i == k_star:
                    continue
                if emp_mean[i] - emp_mean[k_star] >= 2.0 * epsilon:
                    weights = [1.0] * num_actions
                    print("Drift detected")

            pull_cnt = list()
            emp_mean = list()

            for i in range(num_actions):
                pull_cnt.append(0)
                emp_mean.append(0.0)

        yield crnt_loss, weights


def SW_UCB(loss_vector, lm, break_point_list):
    num_actions = lm.get_num_actions()
    num_rounds = lm.get_num_rounds()

    # optimal_action_switch_list holds a list of rounds where the action of benchmark strategy switches
    # In normal case, we'll keep both the lists same
    optimal_action_switch_list = break_point_list
    best_action = list()

    # Here we are computing the benchmark strategy
    start_rnd = 1
    for change_pt in optimal_action_switch_list:
        min_loss_action = min(
            range(num_actions),
            key=lambda action: sum([loss_vector[action][rnd] for rnd in range(start_rnd, change_pt)])
        )
        best_action.append(min_loss_action)
        start_rnd = change_pt + 1

    print(best_action)
    cumulative_loss = 0
    best_action_cumulative_loss = 0
    weak_regret = []

    t = 0
    crnt_int_index = 0
    int_cntr = 0
    hold = optimal_action_switch_list[crnt_int_index]
    # print hold, optimal_action_switch_list

    for (crnt_loss) in sw_ucb_engine(lm, loss_vector):

        if t >= num_rounds - 1:
            break

        if t > hold:
            crnt_int_index += 1
            print(t, crnt_int_index)
            hold = optimal_action_switch_list[crnt_int_index]

        crnt_bestAction = best_action[crnt_int_index]

        cumulative_loss += crnt_loss
        best_action_cumulative_loss += loss_vector[crnt_bestAction][t]

        weak_regret.append(cumulative_loss - best_action_cumulative_loss)
        t += 1

    return weak_regret


def Exp3R(loss_vector, lm, break_point_list, option):
    num_actions = lm.get_num_actions()
    num_rounds = lm.get_num_rounds()
    access_loss = lambda action, rnd: loss_vector[action][rnd]

    optimal_action_switch_list = break_point_list
    best_action = list()

    # computing the benchmark strategy
    start_rnd = 1
    for change_pt in optimal_action_switch_list:
        min_loss_action = min(
            range(num_actions),
            key=lambda action: sum([loss_vector[action][rnd] for rnd in range(start_rnd, change_pt)])
        )
        best_action.append(min_loss_action)
        start_rnd = change_pt + 1

    print(best_action)
    cumulative_loss = 0
    best_action_cumulative_loss = 0
    weak_regret = []

    t = 1
    crnt_int_index = 0
    int_cntr = 0
    hold = optimal_action_switch_list[crnt_int_index]

    for (crnt_loss, weights) in old_exp3_mod_engine(lm, access_loss, option):

        if t >= num_rounds - 1:
            break

        if t > hold:
            crnt_int_index += 1
            print(t, crnt_int_index)
            hold = optimal_action_switch_list[crnt_int_index]

        crnt_best_action = best_action[crnt_int_index]

        cumulative_loss += crnt_loss
        best_action_cumulative_loss += loss_vector[crnt_best_action][t]

        weak_regret.append(cumulative_loss - best_action_cumulative_loss)
        t += 1

    return weak_regret
