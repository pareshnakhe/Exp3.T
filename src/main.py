from prob import *
import numpy as np
import random
import pylab
import threading
from src.loss_monitor import LossMonitor
from src.mab_models import *


def make_bias_list(num_rounds):
	bias = list()
	# Choose the biases of two actions from separate intervals

	bias.append(np.random.uniform(0.5, 0.6, num_rounds))
	bias.append(np.random.uniform(0.40, 0.5, num_rounds))
	return bias


def make_bias_list_switch(lm, input_bias, break_point_list):
	"""
	Generates a list of biases with changes in trend as given by "break_pt_list"
	"""
	num_actions = lm.get_num_actions ()

	# bias() is a list: [ [action 1 losses], [actn 2 losses] ...]
	bias = list ()
	# Create a list of lists trend = [[], [] ...]
	# trend[i] holds biases for the entire time period for action i
	trend = list ()
	for i in range (num_actions):
		trend.append ([])

	base = 0
	temp = 0
	for i in range (len (break_point_list)):
		for j in range (num_actions):
			index = (base + j)%num_actions
			trend[j].extend (
					(np.random.uniform (
							input_bias[index][0], input_bias[index][1], break_point_list[i] - temp
					)).tolist ()
			)
		base = (base + 1)%num_actions
		temp = break_point_list[i]

	for i in range (num_actions):
		bias.append (trend[i])
	return bias


"""
NOTE: Generating ARG losses is difficult since it is difficult to maintain the ARG property
without making up a trivial example. For this reason, we pick loss vales from a truncated
Gaussian distribution and approximate the value of \Delta_{sp} by the gap in expected loss.
"""


def generate_ARG_losses(lm, break_point_list, option):
	num_actions = lm.get_num_actions()

	lossVector = list()
	input_range = list()

	# Setting parameters for input range of each action
	if option == 2:
		mu_base = 0.450
		sigma = 0.05
		input_range.append([0.30, sigma])
	else:
		mu_base = 0.650
		sigma = 0.05
		input_range.append([0.30, sigma])

	for i in range(1, num_actions):
		input_range.append([mu_base + i * 0.010, sigma])

	# **** Delta_{sp} is taken as (mu1 - mu2) / 2
	lm.Delta = (mu_base - input_range[0][0]) / 2.0
	# loss_bound does help us with faster trend detection
	# This is additional information available to us
	lm.loss_bound = 0.4
	print(input_range)

	# Create a list of lists trend = [[], [] ...]
	# trend[i] holds loss values for the entire time period for action i
	trend = list()
	for i in range(num_actions):
		trend.append([])

	base = 0
	temp = 0
	for i in range(len(break_point_list)):
		for j in range(num_actions):
			index = (base + j) % num_actions
			hold = np.random.normal(
					input_range[index][0], input_range[index][1], break_point_list[i] - temp
			).tolist()
			# ensuring that losses are in [0,1] range
			for itr in range(len(hold)):
				if hold[itr] < max(input_range[index][0] - 0.2, 0.0):
					hold[itr] = max(input_range[index][0] - 0.2, 0.0)
				if hold[itr] > min(input_range[index][0] + 0.2, 1.0):
					hold[itr] = min(input_range[index][0] + 0.2, 1.0)

			trend[j].extend(hold)
		base = (base + 1) % num_actions
		temp = break_point_list[i]

	for i in range(num_actions):
		lossVector.append(trend[i])

	return lossVector


def generate_stochastic_losses(lm, break_point_list, option):
	"""
	Generates actual loss vector on which (any) algorithm runs.
	:return: lossVector (lossVector[action][t] returns the loss of action at round t)
	"""

	num_rounds = lm.get_num_rounds ()
	num_actions = lm.get_num_actions ()
	input_bias = list ()

	if option == 0:
		mu_base = 0.450
		# The numbers in input_bias indicate the range of variation of bias for the particular action
		# input_bias.append([0.10, 0.15])
		input_bias.append ([0.10, 0.1001])
	else:
		mu_base = 0.650
		# The numbers in input_bias indicate the range of variation of bias for the particular action
		# input_bias.append([0.10, 0.15])
		input_bias.append ([0.10, 0.1001])

	for i in range (1, num_actions):
		# input_bias.append([mu_base, np.random.uniform(mu_base, mu_base + 0.10)])
		input_bias.append ([mu_base, np.random.uniform (mu_base, mu_base + 0.1001)])

	# Changed from 4.0 to 2.0
	lm.Delta = (mu_base - input_bias[0][0])/4.0
	print (input_bias)

	if break_point_list:
		biasList = make_bias_list_switch (lm, input_bias, break_point_list)
	else:
		biasList = make_bias_list (num_rounds)

	lossVector = list ()
	for bList in biasList:
		lossVector.append ([1 if random.random () < bias else 0 for bias in bList])

	return lossVector


def module(option, action_set):
	for k in action_set:
		print ("With {} actions:".format (k))
		lm = LossMonitor (k, 100000)

		# break_point_list holds a list of rounds where trend changes
		break_point_list = [20000, 42000, lm.get_num_rounds ()]

		if option < 2:
			loss_vector = generate_stochastic_losses (lm, break_point_list, option)
		else:
			loss_vector = generate_ARG_losses(lm, break_point_list, option)

		num_iterations = 1

		for itr in range(num_iterations):
			print("Iteration number: {}".format(itr))
			regret_swucb = SW_UCB(loss_vector, lm, break_point_list)
			lm.clean_object()
			regret_exp3r = Exp3R(loss_vector, lm, break_point_list, option)
			lm.clean_object()
			regret_exp3base = Exp3Switch(loss_vector, lm, break_point_list, old_exp3_engine)
			lm.clean_object()
			regret_exp3s = Exp3Switch(loss_vector, lm, break_point_list, exp3S_engine)
			lm.clean_object()
			regret_exp3d = Exp3D(loss_vector, lm, break_point_list)
			lm.clean_object()

		threadLock.acquire()
		pylab.xlabel("rounds")
		pylab.ylabel("Cumulative Regret")
		pylab.ticklabel_format(style='sci', scilimits=(0, 0), axis='both')

		pylab.plot(regret_swucb, 'm', label='SW-UCB')
		pylab.plot(regret_exp3r, 'c', label='Exp3.R')
		pylab.plot(regret_exp3base, 'b', linewidth=1.0, label='Base Exp3')
		pylab.plot(regret_exp3s, 'r', label='Exp3.S')
		pylab.plot(regret_exp3d, 'g', linewidth=1.0, label='Exp3D')
		pylab.legend(loc='upper left')

		pylab.savefig('TEST: HOption=' + str(option) + ': Fig-' + str(k) + '-actions.png')
		pylab.clf()
		threadLock.release()


class Threaded (threading.Thread):
	def __init__(self, option, action_set):
		threading.Thread.__init__(self)
		self.option = option
		self.action_set = action_set

	def run(self):
		print("Starting " + self.name)
		module(self.option, self.action_set)
		print("Exiting " + self.name)


if __name__ == "__main__":
	action_set = [2]
	threadLock = threading.Lock()
	# Anonymous function to access loss of a given action in a given round
	# accessLoss = lambda action, round: lossVector[action][round]

	thread0 = Threaded(0, action_set)
	thread1 = Threaded(1, action_set)
	thread2 = Threaded(2, action_set)
	thread3 = Threaded(3, action_set)

	thread0.start()
	thread1.start()
	thread2.start()
	thread3.start()
