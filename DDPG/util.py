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

	def push(self, state, action, reward, next_state):
		if len(self.buffer) < self.capacity:
			self.buffer.append(None)
		self.buffer[self.position] = (state, action, reward, next_state)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		batch = random.sample(self.buffer, batch_size)
		#print(f"BATCH: {list(zip(*batch))}")
		#print(f"BATCHlen: {len(list(zip(*batch)))}")
		temp = zip(*batch)
		# for elem in temp:
		# 	print(np.asarray(elem).shape)
		state, action, reward, next_state = map(np.stack, zip(*batch))
		return state, action, reward, next_state

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

# define soft update with parameter tau
def soft_update(model, model_target, tau):
	for target_param, param in zip(model_target.parameters(), model.parameters()):
		target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
		
# define hard update to directly copy parameters
def hard_update(model, model_target):
	soft_update(model, model_target, 1.0)
