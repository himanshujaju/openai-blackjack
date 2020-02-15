import gym
from agent import SarsaAgent

EPISODES = 1000000
EPOCH = 10000

if __name__ == '__main__':
	env = gym.make('Blackjack-v0')
	agent = SarsaAgent(env.observation_space, env.action_space)
	
	win = 0
	loss = 0

	for episode in range(1, EPISODES + 1):
		done = 0
		current_state = env.reset()
		current_action = agent.choose_action(current_state)
		reward = 0

		while not done:
			next_state, reward, done, _ = env.step(current_action)
			next_action = agent.choose_action(next_state)

			agent.update(current_state, current_action, reward, next_state, next_action)

			current_state = next_state
			current_action = next_action

		# Stats computation
		if reward > 0:
			win = win + 1;
		else:
			loss = loss + 1

		# Progress output
		if episode % EPOCH == 0:
			print("------------------")
			print("Episodes done :", episode)
			win_ratio = (1.0 * win) / episode
			loss_ratio = (1.0 * loss) / episode

			print("Won :", win_ratio)
			print("Loss :", loss_ratio)

	print(agent.Q)
	env.close()