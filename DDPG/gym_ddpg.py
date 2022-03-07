# copy of ddpg.py, but for gym environments instead of dm_control
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
import time
import gym

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
num_epochs = 200
max_steps = 250

# main training loop for DDPG
def train(env):
	act_spec = env.action_space # action_spec is a BoundedArray with shape field

	obs_dim  = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]

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

	act_min, act_max = act_spec.low, act_spec.high

	value_criterion = nn.MSELoss()

	def ddpg_update(buffer):
		# sample a batch from the replay buffer
		state, action, reward, next_state, done = buffer.sample(batch_size)

		state = torch.FloatTensor(state)
		action = torch.FloatTensor(action)
		next_state = torch.FloatTensor(next_state)
		reward = torch.FloatTensor(reward).unsqueeze(-1)
		done = torch.FloatTensor(done).unsqueeze(-1)

		with torch.no_grad():
			next_action = actor_target(next_state) # compute with target

			# we detach the next_action, because the only parameters we wish to update are the critic's
			target_value = critic_target(next_state, next_action) # compute with target critic
			expected_value = reward + (1-done) * gamma * target_value #TODO: add done information

		#print(f"Q avg: {torch.mean(target_value)}") # the mean taken over (batch size) examples
		value = critic(state, action)
		q_loss = value_criterion(value, expected_value) # detach from computation graph

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
	eval = False # change render and eval to true when testing learned policy
	avg_ep_rewards, q_losses, pi_losses = [], [], []
	while steps < num_epochs * steps_per_epoch:
		state = env.reset()
		noise.reset()
		# state = obs_to_tensor(time_step.observation)

		if steps + max_steps >= num_epochs * steps_per_epoch: # render the last ep; remove noise
			print("Rendering the following episode")
			eval = True

		episode_reward, action_list = [], []
		for step in range(max_steps): # start a new episode
			state = torch.FloatTensor(state)
			action = actor(state).detach()
			action_list.append(action)
			action = denormalize_actions(action, act_min, act_max)
			if not eval: #TODO: decrease noise over time
				action = noise.get_action(action, steps) # inject OU noise (decaying over time)
			next_state, reward, done, _ = env.step(action)
			
			episode_reward.append(reward)

			buffer.push(state, action, reward, next_state, done)
			if len(buffer) > batch_size: # update if we have enough samples
				losses = ddpg_update(buffer)
				q_losses.append(losses[0])
				pi_losses.append(losses[1]);

			state = next_state
			steps += 1

			if steps % steps_per_epoch == 0:
				epoch_num = steps // steps_per_epoch
				print(f"Epoch number {epoch_num}")
				print(f"q_loss: {sum(q_losses[-10:])/10}, pi_loss: {sum(pi_losses[-10:])/10}")
				print(f"average reward per timestep: {sum(episode_reward)/len(episode_reward)}")
				fig, axs = plt.subplots(2, 1)
				axs[0].plot(np.arange(len(action_list)), action_list)
				axs[1].plot(np.arange(len(episode_reward)), episode_reward)
				plt.savefig(f"./rewards/report{epoch_num}.png")
				time.sleep(5)

			if done: # if gym episode terminates on this step
				break

		avg_ep_rewards.append(sum(episode_reward)/len(episode_reward))
		recorder.save_rewards(episode_reward)
	print("training complete; rendering video of policy from frames...")
	fig, axs = plt.subplots(2, 2)

	# X axis length may differ slightly since 64 samples must be collected before ddpg_update
	axs[0][0].plot(np.arange(len(episode_reward)), episode_reward)
	axs[0][1].plot(np.arange(len(avg_ep_rewards)),avg_ep_rewards)
	axs[1][0].plot(np.arange(len(q_losses)), q_losses)
	axs[1][1].plot(np.arange(len(pi_losses)), pi_losses)
	plt.savefig("training.png")
	torch.save(actor.state_dict(), "./saved_models/actor.pt")
	torch.save(critic.state_dict(), "./saved_models/critic.pt")
	#recorder.render_video()

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

if __name__ == '__main__':
	print("Training DDPG with default hyperparameters")
	env_name = "Pendulum-v0"
	env = gym.make(env_name)
	env = gym.wrappers.Monitor(env, "recording",force=True)
	train(env)
