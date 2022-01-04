# Implementation of PPO
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from torch.distributions import Categorical

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

# class with lists containing actions, states, rewards, etc.
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.is_terminals = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.is_terminals.clear()

class PPO:
    def __init__(self, env):
        self.action_dim = env.action_space.n
        self.state_dim = env.observation_space.shape[0]

        print(f"action: {self.action_dim}, state: {self.state_dim}")

        self.eps = 0.2
        self.gamma = 0.99
        self.max_ep_length = 1000
        self.samples_per_policy = self.max_ep_length * 4
        self.max_time_steps = 400000
        self.num_epochs = 80
        self.hidden_size = 16
        self.render = True

        self.data = RolloutBuffer()
        self.critic = Critic(self.state_dim, self.hidden_size)
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_size)

        lr_actor = 0.0003       # learning rate for actor network
        lr_critic = 0.001       # learning rate for critic network

        self.optimizer = torch.optim.Adam([
                        {'params': self.actor.parameters(), 'lr': lr_actor},
                        {'params': self.critic.parameters(), 'lr': lr_critic}
                    ])
        self.MseLoss = nn.MSELoss()

    def train(self):
        # Collect trajectories using policy Theta_k
        time_step = 0
        episode = 0
        while time_step < self.max_time_steps:
            state = env.reset()

            if episode % (100) == 0:
                self.render = True
            else:
                self.render = False
            episode += 1

            for i in range(self.max_ep_length):
                action = self.select_action(state)
                new_state, reward, done, _ = env.step(action)
                time_step += 1

                if self.render:
                    env.render()
                    time.sleep(0.01)

                # add data
                self.data.rewards.append(reward)
                self.data.is_terminals.append(done)

                # update if needed
                if time_step % self.samples_per_policy == 0:
                    self.update()

                # print("Reward:", reward)

                if done:
                    break

                state = new_state
            print("max episode step reached:", i)

    def select_action(self, state):
        # use current policy to sample action
        state = torch.FloatTensor(state)
        actor_output = self.actor(state)
        dist = Categorical(actor_output)
        action = dist.sample()

        # these data must be added as tensors
        self.data.states.append(state)
        self.data.actions.append(action)
        self.data.log_probs.append(dist.log_prob(action))
        return action.item()

    def update(self):
        # compute rewards to go
        # zip data rewards and is_terminals together
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.data.rewards), reversed(self.data.is_terminals)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            # print("reward", reward, " disc", discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32)

        rewards = (rewards - rewards.mean()) / rewards.std()

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.data.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.data.actions, dim=0)).detach()
        old_log_probs = torch.squeeze(torch.stack(self.data.log_probs, dim=0)).detach()

        # evaluate a new policy at the old transitions and take a opt step using gradient of loss

        for i in range(self.num_epochs):
            log_probs, state_values = self.evaluate(old_states, old_actions)

            ratios = torch.exp(log_probs - old_log_probs)

            # advantages
            state_values = torch.squeeze(state_values)
            advantages = rewards - state_values

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps, 1+self.eps) * advantages

            if i == 40:
                print(torch.min(surr1, surr2).mean(), self.MseLoss(rewards, state_values))
            loss = -1 * torch.min(surr1, surr2) + 0.5 * self.MseLoss(rewards, state_values)
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        # self.plot_visualizations()
        self.data.clear()
        print("done update")

    def plot_visualizations(self):
        # use matplotlib to show action dist
        n, bins, patches = plt.hist(self.data.actions, self.action_dim, facecolor='blue', alpha=0.5)
        plt.show()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        state_values = self.critic(state)
        return action_logprobs, state_values

if __name__ == "__main__":
    env = gym.make("MountainCar-v0").env

    ppo = PPO(env)
    ppo.train()
