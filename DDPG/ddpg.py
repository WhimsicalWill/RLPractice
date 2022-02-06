from util import Actor, Critic, ReplayBuffer
from util import hard_update, soft_update
import numpy as np
from dm_control import suite
import torch.optim as optim
import torch.nn as nn
import torch
from torch.distributions import Normal

# hyperparameters for DDPG implementation
pi_lr = 1e-3
q_lr = 1e-4 # critic lr is smaller, so receives 10x smaller changes per step
capacity = 10000
gamma = 0.99
tau = 0.01
batch_size = 8
# steps_per_update = 20
steps_per_epoch = 1000
num_epochs = 10
max_steps = 100

# main training loop for DDPG
# TODO: sample env and record trajectories
def train(env):
	act_spec = env.action_spec() # action_spec is a BoundedArray with shape field
	obs_spec = env.observation_spec() # obs_spec is a dict type
	
	obs_dim = get_obs_shape(obs_spec) # get obs space from dm
	act_dim = act_spec.shape[0] # get action dim from dm
	
	h_dim = 16
	actor = Actor(obs_dim, act_dim, h_dim)
	critic = Critic(obs_dim, h_dim)
	actor_target = Actor(obs_dim, act_dim, h_dim)
	critic_target = Critic(obs_dim, h_dim)
	
	# copy initial weights to target models
	hard_update(critic, critic_target)
	hard_update(actor, actor_target)
	
	critic_optimizer = optim.Adam(critic.parameters(), lr=q_lr)
	actor_optimizer = optim.Adam(actor.parameters(), lr=pi_lr)
	
	buffer = ReplayBuffer(capacity)	
	
	act_min, act_max = act_spec.minimum, act_spec.maximum
	
	value_criterion = nn.MSELoss()
	
	steps = 0
	while steps < num_epochs * steps_per_epoch:
		# reset stuff before next episode
		time_step = env.reset()
		state = time_step.observation
		
		for _ in range(max_steps):
			#action = actor(state)
			action = np.random.uniform(act_min, act_max)
			time_step = env.step(action)
			
			# TODO: create wrapper for state to tensor; sample actions from actor
			buffer.push(state, action, time_step.reward, time_step.observation)
			if len(buffer) > batch_size: # update if we have enough samples
				ddpg_update(buffer)
			
			state = time_step.observation
			steps += 1
			
			if steps % steps_per_epoch == 0:
				epoch_num = steps // steps_per_epoch
				print(f"Epoch number {epoch_num}")
	
def get_obs_shape(obs_spec):
	result_dim = 0
	for value in obs_spec.values():
		result_dim += value.shape[0]
	return result_dim	
	
def ddpg_update(buffer):
	# sample a batch from the replay buffer
	state, action, reward, next_state = buffer.sample(batch_size)
	# convert to FloatTensor
	# state = torch.FloatTensor(state)
	# next_state = torch.FloatTensor(next_state)
	# action = torch.FloatTensor(action)
	# reward = torch.FloatTensor(reward)
	
	# take a step to maximum using gradient ascent
	

if __name__ == '__main__':
    print("Training DDPG with default hyperparameters")
    env = suite.load("cheetah", "run")
    train(env)
