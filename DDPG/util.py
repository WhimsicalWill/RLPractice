import random
import numpy as np
import torch.nn as nn

class ReplayBuffer:
	def __init__(self, capacity):
		self.capacity = capacity
		self.buffer = []
		self.position = 0

	def push(self, state, action, reward, next_state):
		if len(self.buffer) < self.capacity:
		    self.buffer.append(None)
		self.buffer[self.position] = (state, action, reward, next_state)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		batch = random.sample(self.buffer, batch_size)
		state, action, reward, next_state = map(np.stack, zip(*batch))
		return state, action, reward, next_state

	def __len__(self):
        	return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = self.fc1(state)
        x = self.fc2(F.relu(x))
        x = self.fc3(F.relu(x))
        # softmax output to ensure sum of probabilities is one
        return self.softmax(x)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = self.fc1(state)
        x = self.fc2(F.relu(x))
        x = self.fc3(F.relu(x))
        return x

# define soft update with parameter tau
def soft_update(model, model_target, tau):
	for target_param, param in zip(model_target.parameters(), model.parameters()):
		target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
		
# define hard update to directly copy parameters
def hard_update(model, model_target):
	soft_update(model, model_target, 1.0)
