import math

class LossMonitor:

	def __init__(self, num_actions, num_rounds):

		#primary data structures used
		self.history = [[], [], []]
		self.crnt_interval_loss = [0] * num_actions
		self.crnt_interval_index = 0

		#Game parameters
		self.K = num_actions
		self.T = num_rounds
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

	def get_num_actions(self):
		return self.K

	def get_num_rounds(self):
		return self.T

	def get_global_reset(self):
		return self.global_reset

	def set_global_reset(self, val):
		self.global_reset = val

	def get_kt_star(self):
		delta = math.sqrt((self.K / (self.T * math.log(self.K))))
		a = math.log(2.0 * self.K / delta)
		#B = 2 * math.pow(self.Delta, 2)
		b = min(((8.0 * math.pow(self.Delta, 2)) / math.pow(self.loss_bound, 2.0)), 1)
		return int(self.K * (a / b))

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
		print("interval length = {}".format(inter_len))

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
		print("Reset")
		# global_reset is a flag for Bubeck_exp3_mod to restart
		self.set_global_reset(1)
		self.reset_hold = 1

	def trend_detection(self):
		#'flag' is an indicator variable to check if exp3 was reseted in this call
		flag = 0
		t_star = int(self.get_kt_star() / self.K)
		self.history[self.crnt_interval_index % 3] = [float(loss) / t_star for loss in self.crnt_interval_loss]

		if self.reset_hold == 1:
			print("trend detection skipped")
			self.reset_hold = 0
		else:
			print("trend detection running")

			if self.history[0] and self.history[1] and self.history[2]:
				min_loss_action_crnt = min(range(self.K), key=lambda action: self.history[self.crnt_interval_index % 3][action])
				min_loss_action_old = min(range(self.K), key=lambda action: self.history[(self.crnt_interval_index + 1) % 3][action])

				if min_loss_action_crnt != min_loss_action_old:
					self.reset_exp3()
					flag = 1

		self.crnt_interval_index += 1
		self.crnt_interval_loss = [0] * self.K
		return flag
