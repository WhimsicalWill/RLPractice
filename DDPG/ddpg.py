from util import Actor, Critic, ReplayBuffer
import numpy as np
from dm_control import suite
import torch.optim as optim
import torch
from torch.distributions import Normal

# hyperparameters for DDPG implementation
pi_lr = 1e-3
q_lr = 1e-4 # critic lr is smaller, so receives 10x smaller changes per step
capacity = 10000
gamma = 0.99
tau = 0.01
batch_size = 16
steps_per_update = 20
steps_per_epoch = 1000

# main training loop for DDPG
# TODO: sample env and record trajectories
def train(env):
	# initialize models and target models
	obs_dim = 1 # get obs space from dm
	act_dim = 1 # get action dim from dm
	h_dim = 16
	
	actor = Actor(obs_dim, act_dim, h_dim)
	critic = Critic(obs_dim, h_dim)
	
	actor_target = Actor(obs_dim, act_dim, h_dim)
	critic_target = Critic(obs_dim, h_dim)
	
	# transfer weights with an initial "hard" update
	
	critic_optimizer = optim.Adam(critic.parameters(), lr=q_lr)
	actor_optimizer = optim.Adam(actor.parameters(), lr=pi_lr)
	
	buffer = ReplayBuffer(capacity)	
	
	action_spec = env.action_spec()
	act_min, act_max = action_spec.minimum, action_spec.maximum
	
	
	time_step = env.reset()
	state = time_step.observation
	counter = 0
	for time_step in range(steps_per_epoch): # collect one epoch worth of samples
		random_action = np.random.uniform(act_min, act_max)
		time_step = env.step(random_action)
		buffer.push(state, random_action, time_step.reward, time_step.observation)
		state = time_step.observation
		if counter % 100 == 0:
			print("Reward: " + str(time_step.reward))
		counter += 1
	
	print(buffer.sample(batch_size))
	
def ddpg_update():
	# sample a batch from the replay buffer
	state, action, reward, next_state, done = replay_buffer.sample(batch_size)
	# convert to FloatTensor
	state = torch.FloatTensor(state)
	next_state = torch.FloatTensor(next_state)
	action = torch.FloatTensor(action)
	reward = torch.FloatTensor(reward)
	done = torch.FloatTensor(torch.float32(done))
	
	# take a step to maximum using gradient ascent
	

if __name__ == '__main__':
    print("Training DDPG with default hyperparameters")
    env = suite.load("cheetah", "run")
    train(env)
