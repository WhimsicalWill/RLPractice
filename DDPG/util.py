import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReplayBuffer:
	def __init__(self, capacity):
		self.capacity = capacity
		self.buffer = []
		self.position = 0

	def push(self, state, action, reward, next_state, done):
		if len(self.buffer) < self.capacity:
			self.buffer.append(None)
		self.buffer[self.position] = (state, action, reward, next_state, done)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		batch = random.sample(self.buffer, batch_size)
		state, action, reward, next_state, done = map(np.stack, zip(*batch))
		return state, action, reward, next_state, done

	def __len__(self):
        	return len(self.buffer)

# the actor class takes a state and outputs the predicted best action
# the input has state shape and output should have
# action shape with each value inbetween lower and upper action bounds
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = self.fc1(state)
        x = self.fc2(F.relu(x))
        x = self.fc3(F.relu(x))
        # tanh gives action values between -1 and 1 (normalized)
        return torch.tanh(x)

class Critic(nn.Module):
	def __init__(self, state_dim, act_dim, hidden_size):
		super(Critic, self).__init__()
		self.fc1 = nn.Linear(state_dim + act_dim, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, 1)

	def forward(self, state, action):
		x = torch.cat((state, action), dim=-1) # concat the two inputs
		x = self.fc1(x)
		x = self.fc2(F.relu(x))
		x = self.fc3(F.relu(x))
		return x

class OUNoise:
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
		# with default params, sigma never changes
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

# define soft update with parameter tau; copy model params into model_target
def soft_update(model, model_target, tau):
	for target_param, param in zip(model_target.parameters(), model.parameters()):
		target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
		
# define hard update to directly copy parameters
def hard_update(model, model_target):
	soft_update(model, model_target, 1.0)
