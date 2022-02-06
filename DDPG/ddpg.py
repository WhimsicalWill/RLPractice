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
	critic = Critic(obs_dim, act_dim, h_dim)
	actor_target = Actor(obs_dim, act_dim, h_dim)
	critic_target = Critic(obs_dim, act_dim, h_dim)
	
	# copy initial weights to target models
	hard_update(critic, critic_target)
	hard_update(actor, actor_target)
	
	critic_optimizer = optim.Adam(critic.parameters(), lr=q_lr)
	actor_optimizer = optim.Adam(actor.parameters(), lr=pi_lr)
	
	buffer = ReplayBuffer(capacity)	
	
	act_min, act_max = act_spec.minimum, act_spec.maximum
	
	value_criterion = nn.MSELoss()
	
	def ddpg_update(buffer):
		# sample a batch from the replay buffer
		state, action, reward, next_state = buffer.sample(batch_size)
		# print("DDPG Update")

		state = torch.FloatTensor(state)
		action = torch.FloatTensor(action)
		next_state = torch.FloatTensor(next_state)
		reward = torch.FloatTensor(reward)

		# freeze the target q networks weights for backprop
		# is this even necessary if we only compute gradients for actor/critic
		next_action = actor_target(next_state) # compute with target

		# we detach the next_action, because the only parameters we wish to update are the critic's
		target_value = torch.unsqueeze(reward, dim=-1) + gamma * critic_target(next_state, next_action.detach()) # compute with target
		value = critic(state, action)
		q_loss = value_criterion(value, target_value)

		pi_loss = critic(state, actor(state))
		pi_loss = -pi_loss.mean() # optim defaults to gradient descent; transform to negative scalar

		critic_optimizer.zero_grad()
		q_loss.backward()
		critic_optimizer.step()

		actor_optimizer.zero_grad()
		pi_loss.backward()
		actor_optimizer.step()

		# do soft updates on the target actor and critic models
		soft_update(actor, actor_target, tau) # copy a fraction of actor weights into actor_target
		soft_update(critic, critic_target, tau) # copy a fraction of critic weights into critic_target

		return q_loss, pi_loss # return the losses in case we want to track our losses

	steps = 0
	while steps < num_epochs * steps_per_epoch:
		# reset stuff before next episode
		time_step = env.reset()
		state = obs_to_tensor(time_step.observation)
		
		for step in range(max_steps):
			#action = actor(state)
			action = np.random.uniform(act_min, act_max)
			time_step = env.step(action)
			next_state = obs_to_tensor(time_step.observation)

			buffer.push(state, action, time_step.reward, next_state)
			if len(buffer) > batch_size: # update if we have enough samples
				losses = ddpg_update(buffer)
			
			state = next_state
			steps += 1
			
			if steps % steps_per_epoch == 0:
				epoch_num = steps // steps_per_epoch
				print(f"Epoch number {epoch_num}")
				print(losses)

def obs_to_tensor(observation):
	obs_tensor = []
	for array in observation.values():
		obs_component = torch.as_tensor(array).float() # uses 32 bit float precision
		# unsqueeze scalars into 1 by 1 tensors
		if len(obs_component.shape) == 0:
			obs_component = torch.unsqueeze(obs_component, 0)
		obs_tensor.append(obs_component) # append a component of the obs to the final tensor
	return torch.cat(obs_tensor, dim=-1) # horizontal stack (column wise)
	
def get_obs_shape(obs_spec):
	result_dim = 0
	# print(obs_spec)
	for value in obs_spec.values():
		if len(value.shape) == 0:
			result_dim += 1
		else:
			result_dim += value.shape[0]
	print(result_dim)
	return result_dim	

if __name__ == '__main__':
    print("Training DDPG with default hyperparameters")
    env = suite.load("walker", "walk")
    train(env)
