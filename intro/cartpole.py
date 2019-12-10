# A plain Cartpole with a random action wrapper


import gym
import random


class RandomActionWrapper(gym.ActionWrapper):

	def __init__(self, env, epsilon=0.1):
		super(RandomActionWrapper,self).__init__(env)
		self.epsilon = epsilon

	def action(self, action):
		if random.random() < self.epsilon:
			print('Random!')
			return self.env.action_space.sample()

		return action

if __name__ == "__main__":
	# env = gym.make("CartPole-v0")
	env = RandomActionWrapper(gym.make("CartPole-v0"))
	env = gym.wrappers.Monitor(env, "recording", force=True)
	total_reward = 0.0
	total_steps = 0
	obs = env.reset()

while True:
	action = 0
	obs, reward, done, _ = env.step(action) # _ is 'info' - can be useful for debugging, but the agent is not allowed to use it
	
	total_reward += reward
	total_steps += 1
	if done:
		break


print("Episode done in %d steps with reward of %.2f" % (total_steps, total_reward))

env.env.close()
