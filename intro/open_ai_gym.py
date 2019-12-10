import random

class Environment:

	def __init__(self):
		self.steps_left = 10

	def get_observation(self):
		return [0.0, 0.0, 0.0, 0.0]

	def get_actions(self):
		return [0, 1]

	def is_done(self):
		return self.steps_left == 0

	'''
	action() handles the agent's action and returns the reward for that action
	'''
	def action(self, action):
		if self.is_done():
			raise Exception("Game is over")
		
		self.steps_left -= 1
		return random.random()


class Agent:
	
	def __init__(self):
		self.total_reward = 0.0

	'''
	step() accepts the environment as an argument and allows the agent to:
		- observe the environment
		- make a decision about the action to take, based on the observations
		- submit the action to the environment
		- get the reward for the current step
	'''
	def step(self, env):
		current_obs = env.get_observation()
		actions = env.get_actions()
		reward = env.action(random.choice(actions))
		self.total_reward += reward


if __name__ == "__main__":
	env = Environment()
	agent = Agent()

	while not env.is_done():
		agent.step(env)

	print("Total reward:", agent.total_reward)





