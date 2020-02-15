import numpy as np
from collections import defaultdict

# Parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 0.2

class SarsaAgent(object):
    def __init__(self, observation_space, action_space):
    	self.action_space = action_space
    	self.Q = defaultdict(lambda: np.zeros(action_space.n))

    # Epsilon greedy
    def choose_action(self, state):
    	if np.random.uniform(0, 1) < EPSILON :
    		return self.action_space.sample()
    	else:
    		return np.argmax(self.Q[state])

    def update(self, current_state, current_action, reward, next_state, next_action):
    	predicted = self.Q[current_state][current_action]
    	target = reward + DISCOUNT_FACTOR * self.Q[next_state][next_action]

    	self.Q[current_state][current_action] = self.Q[current_state][current_action] + LEARNING_RATE * (target - predicted)
    	return