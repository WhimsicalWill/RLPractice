import torch
import numpy as np

# Replay Buffer that supports clearing, adding, and has max_size
class ReplayBuffer():
	def __init__(self, state_dimension, action_dimension, max_size):
		self.state_dim = state_dimension
		self.action_dim = action_dimension
		self.max_size = max_size
		self.clear() # create replay buffer objects
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		
	def add(self, state, action, next_state, reward, done):
		self.state[self.index] = state
		self.action[self.index] = action
		self.next_state[self.index] = next_state
		self.reward[self.index] = reward
		self.is_terminal[self.index] = done
		self.index = (self.index + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)
	
	# sample a batch of a certain size from ReplayBuffer
	def sample(self, batch_size):
	
	def clear(self):
		self.state = np.zeros((self.max_size, self.state_dim))
		self.action = np.zeros((self.max_size, self.action_dim))
		self.next_state = np.zeros((self.max_size, self.state_dim))
		self.reward = np.zeros((self.max_size, 1))
		self.is_terminal = np.zeros((self.max_size, 1))
		self.index = 0;
		self.size = 0;
