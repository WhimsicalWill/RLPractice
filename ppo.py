# Implementation of PPO
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(ActorCritic, self).__init__()
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

class PPO:
    def __init__(self):


if __name__ == "__main__":
    # run PPO directly

    # 1) Collect trajectories until bucket full
    # 2) Calculate ratio r(pi) of log probs of actions
    # 3) Use torch to maximize surrogate loss
    # 4) Update actor with multiple runs of SGD
    # 5) Fit value function (critic) using samples
    # Note: Critic (value function) is needed for GAE
