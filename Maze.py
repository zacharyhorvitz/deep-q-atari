import tensorflow as tf
import numpy as np
import copy

class Maze(object):
	'''
	A class representing a 10 x 10 2D MNIST Maze
	'''
	def __init__(self, wall_density=0.1):
		'''

		:param wall_density: the probability of a wall at any location in the maze
		'''
		self.height = 10
		self.width = 10
		self.goal_state = np.array([9, 9])
		self.wall_density = wall_density
		self.state = (0, 0)
		# a 2D binary array representing wall locations (1 = wall, 0 = no wall)
		self.walls = np.vectorize(lambda r: 1 if r < self.wall_density else 0)\
			(np.random.rand(self.height, self.width))
		# MNIST data
		self.walls[0, 0], self.walls[9, 9] = 0, 0
		(train_images, train_labels), (test_images, test_labels) = \
			tf.keras.datasets.mnist.load_data()
		self.x_trains = np.array([train_images[train_labels == i] for i in range(10)])
		self.x_tests = np.array([test_images[test_labels == i] for i in range(10)])

	def get_state_mnist(self, state):
		'''

		:param state: a tuple representing the coordinates in the maze (x, y) where 0 <= x, y <= 9
		:return: a 2-element numpy array of MNIST digits corresponding to the state passed in
		'''
		return np.array([np.random.choice(self.x_trains[state[0]]), np.random.choice(self.x_trains[state[1]])])

	def reset(self):
		self.state = (0, 0)
		self.state_mnist = self.get_state_mnist(self.state)

	def act(self, action):
		'''

		:param action: 0 (up), 1 (right), 2 (down), or 3 (left)
		:return: a 3-tuple containing the MNIST pair representing the state after acting,
		the reward after acting, and a boolean indicating whether the goal state has been reached
		'''
		next_state = copy.copy(self.state)
		if action == 0:
			next_state[1] = min(next_state[1] + 1, 9)
		elif action == 1:
			next_state[0] = min(next_state[0] + 1, 9)
		elif action == 2:
			next_state[1] = max(next_state[1] - 1, 0)
		elif action == 3:
			next_state[0] = max(next_state[0] - 1, 0)
		else:
			raise ValueError('Invalid Action')
		if self.walls[next_state] == 1:
			reward = -1.0
			reached_goal = False
		else:
			self.state = copy.copy(next_state)
			if np.array_equal(self.state, self.goal_state):
				reward = 1000.0
				reached_goal = True
			else:
				reward = 0.0
				reached_goal = False
		return self.get_state_mnist(self.state), reward, reached_goal

class Maze_Stochasic(Maze):
	'''
	A class representing the same 2D MNIST maze environment above but with stochasticity
	'''
	def __init__(self, wall_density=0.1, stochasticity=0.05):
		'''

		:param wall_density: the probability of a wall at any location in the maze
		:param stochasticity: the probability that a random action is selected when the act() method is called
		'''
		super().__init__(wall_density)
		self.stochasticity = stochasticity

	def act(self, action):
		next_state = copy.copy(self.state)
		if np.random.rand() < self.stochasticity:
			action = np.random.choice([0, 1, 2, 3])
		if action == 0:
			next_state[1] = min(next_state[1] + 1, 9)
		elif action == 1:
			next_state[0] = min(next_state[0] + 1, 9)
		elif action == 2:
			next_state[1] = max(next_state[1] - 1, 0)
		elif action == 3:
			next_state[0] = max(next_state[0] - 1, 0)
		else:
			raise ValueError('Invalid Action')
		if self.walls[next_state] == 1:
			reward = -1.0
			reached_goal = False
		else:
			self.state = copy.copy(next_state)
			if np.array_equal(self.state, self.goal_state):
				reward = 1000.0
				reached_goal = True
			else:
				reward = 0.0
				reached_goal = False
		return self.get_state_mnist(self.state), reward, reached_goal
