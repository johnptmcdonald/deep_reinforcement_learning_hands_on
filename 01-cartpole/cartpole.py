#!/usr/bin/env python3

# This uses a simple cross-entropy method 
# - i.e. it is like a multi-class classifier where it classifies each state as a particular output action

#  It works in the following way:

# 1 - First, we set up the neural net class (Net) and two named tuples; Episode and EpisodeStep. 
	# - The Episode tuple consists of 'reward' (an integer of the reward from an episode) and 'steps' (a list of EpisodeSteps). 
	# - The EpisodeStep tuple consists of an 'observation' (the state) and the action taken in that state (the single integer action)

# 2 - Generate data by navigating the environment (the function 'iterate_batches').

# 3 - Filter the data (the function 'filter_batch') by discarding any episode with a reward below the given percentile. This returns the training set of episodes - good runs where the Net had good actions for each state. 

# 4 - Train the net on these good episodes. We do this by passing in the observations of state from the good episodes and asking the net to predict the correct actions. We then calculate the loss by comparing the Net's predicted actions with what it actually did in the good episode. We then backprop this loss. 

# 5 - repeat steps 2-4

import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


class Net(nn.Module):
	def __init__(self, obs_size, hidden_size, n_actions):
		super(Net, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(obs_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, n_actions)
		)

	def forward(self, x):
		return self.net(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

# creates batches of episodes of the form:
# batches = [Episode(reward: 34, steps: [EpisodeSteps])]
# i.e. a list of episodes, where each episode has the reward and a list  of EpisodeSteps.
# Each EpisodeStep has the observation and the acton taken.
def iterate_batches(env, net, batch_size):
	batch = []
	episode_reward = 0.0
	episode_steps = []
	obs = env.reset()
	sm = nn.Softmax(dim=1)
	while True:
		# print("\n")
		obs_v = torch.FloatTensor([obs])
		# print('Observation', obs)
		# print('Observation Tensor', obs_v)
		# print('Net output', net(obs_v))
		act_probs_v = sm(net(obs_v))
		# print('Softmaxed output', act_probs_v)
		act_probs = act_probs_v.data.numpy()[0]
		# print('Numpy Softmaxed output', act_probs)
		action = np.random.choice(len(act_probs), p=act_probs)
		next_obs, reward, is_done, _ = env.step(action)
		episode_reward += reward
		episode_steps.append(EpisodeStep(observation=obs, action=action))
		if is_done:
			batch.append(Episode(reward=episode_reward, steps=episode_steps))
			episode_reward = 0.0
			episode_steps = []
			next_obs = env.reset()
			if len(batch) == batch_size:
				# print('batch', batch)
				yield batch
				batch = []
		obs = next_obs


def filter_batch(batch, percentile):
	rewards = list(map(lambda s: s.reward, batch))
	# print('rewards', rewards)
	reward_bound = np.percentile(rewards, percentile)
	reward_mean = float(np.mean(rewards))

	train_obs = []
	train_act = []
	for example in batch:
		if example.reward < reward_bound:
			continue
		train_obs.extend(map(lambda step: step.observation, example.steps))
		train_act.extend(map(lambda step: step.action, example.steps))

	# print('train_obs', train_obs)
	# print('train_act', train_act)
	train_obs_v = torch.FloatTensor(train_obs)
	train_act_v = torch.LongTensor(train_act)
	return train_obs_v, train_act_v, reward_bound, reward_mean
	

if __name__ == "__main__":
	env = gym.make("CartPole-v0")
	# env = gym.wrappers.Monitor(env, directory="mon", force=True)
	obs_size = env.observation_space.shape[0]
	n_actions = env.action_space.n

	net = Net(obs_size, HIDDEN_SIZE, n_actions)
	objective = nn.CrossEntropyLoss()
	optimizer = optim.Adam(params=net.parameters(), lr=0.01)
	writer = SummaryWriter(comment="-cartpole")

	for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
		obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
		# print('obs_v', obs_v)
		# print('acts_v', acts_v)
		# print('reward_b', reward_b)
		# print('reward_m', reward_m)

		optimizer.zero_grad()
		action_scores_v = net(obs_v) #i.e. what the network thinks it should do
		# print('action_scores_v', action_scores_v)
		loss_v = objective(action_scores_v, acts_v) # compare it with what it should actually do, as recommend by the previous 'good' episodes
		# print('compare action_scores_v with acts_v', action_scores_v, acts_v)
		# print('loss_v', loss_v)
		loss_v.backward()
		optimizer.step()
		print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (iter_no, loss_v.item(), reward_m, reward_b))
		writer.add_scalar("loss", loss_v.item(), iter_no)
		writer.add_scalar("reward_bound", reward_b, iter_no)
		writer.add_scalar("reward_mean", reward_m, iter_no)
		if reward_m > 199:
			print("Solved!")
			break
	writer.close()
	

