from util import Actor, Critic, ReplayBuffer, OUNoise
from util import hard_update, soft_update
from recorder import Recorder
import numpy as np
from dm_control import suite
import torch.optim as optim
import torch.nn as nn
import torch
from torch.distributions import Normal
import matplotlib.pyplot as plt

# hyperparameters for DDPG implementation
pi_lr = 1e-3
q_lr = 1e-4 # critic lr is smaller, so receives 10x smaller changes per step
capacity = 1000
gamma = 0.99
tau = 0.01
batch_size = 64
h_dim = 64
# steps_per_update = 20
steps_per_epoch = 1000
num_epochs = 300
max_steps = 250 

# main training loop for DDPG
def train(env):
	act_spec = env.action_spec() # action_spec is a BoundedArray with shape field
	obs_spec = env.observation_spec() # obs_spec is a dict type
	
	obs_dim = get_obs_shape(obs_spec) # get obs space from dm
	act_dim = act_spec.shape[0] # get action dim from dm

	print(f"Learning in env with obs_dim: {obs_dim} & act_dim: {act_dim}")
	
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
	noise = OUNoise(act_spec)
	recorder = Recorder(env)
	recorder.clear_directory()
	
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

		with torch.no_grad():
			# freeze the target q networks weights for backprop
			next_action = actor_target(next_state) # compute with target

			# we detach the next_action, because the only parameters we wish to update are the critic's
			target_value = torch.unsqueeze(reward, dim=-1) + gamma * critic_target(next_state, next_action.detach()) # compute with target
		
		value = critic(state, action)
		q_loss = value_criterion(value, target_value) # detach from computation graph

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

		return q_loss.item(), pi_loss.item() # return the losses in case we want to track our losses

	steps = 0
	eval, render = False, False # change to true when testing learned policy
	avg_ep_rewards, q_losses, pi_losses = [], [], []
	while steps < num_epochs * steps_per_epoch:
		# reset stuff before next episode
		time_step = env.reset()
		state = obs_to_tensor(time_step.observation)

		if steps + max_steps >= num_epochs * steps_per_epoch: # render the last ep; remove noise
			print("Rendering the following episode")
			eval = True
			render = True
		
		episode_reward = []
		for step in range(max_steps):
			action = actor(state).detach()
			# TODO: collect actions and visualize distribution over time

			action = denormalize_actions(action, act_min, act_max)
			if not eval: 
				action = noise.get_action(action, steps) # inject OU noise (decaying over time)
			time_step = env.step(action)
			next_state = obs_to_tensor(time_step.observation)
			episode_reward.append(time_step.reward)

			buffer.push(state, action, time_step.reward, next_state)
			if len(buffer) > batch_size: # update if we have enough samples
				losses = ddpg_update(buffer)
				q_losses.append(losses[0])
				pi_losses.append(losses[1]);
			
			if render: # render the env and save
				recorder.save_frame();

			state = next_state
			steps += 1
			
			if steps % steps_per_epoch == 0:
				epoch_num = steps // steps_per_epoch
				print(f"Epoch number {epoch_num}")
				print(losses)
		avg_ep_rewards.append(sum(episode_reward)/len(episode_reward))
	print("training complete; rendering video of policy from frames...")
	fig, axs = plt.subplots(2, 2)
	axs[0][0].plot(np.arange(len(episode_reward)), episode_reward)
	axs[0][1].plot(np.arange(len(avg_ep_rewards)),avg_ep_rewards)
	axs[1][0].plot(np.arange(len(q_losses)), q_losses)
	axs[1][1].plot(np.arange(len(pi_losses)), pi_losses)
	plt.savefig("training.png")
	plt.show()
	# torch.save()
	recorder.render_video()

def denormalize_actions(action, act_min, act_max):
	# shift the range of action from [-1, 1] to [act_min, act_max]
	act_min = torch.tensor(act_min)
	act_max = torch.tensor(act_max)
	result = act_min + ((action + 1) / 2.0) * (act_max - act_min)
	return result.numpy() # cast from torch tensor to nparray for dm_control

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
	for value in obs_spec.values():
		if len(value.shape) == 0:
			result_dim += 1
		else:
			result_dim += value.shape[0]
	return result_dim	

if __name__ == '__main__':
    print("Training DDPG with default hyperparameters")
    env = suite.load("cartpole", "swingup")
    train(env)
