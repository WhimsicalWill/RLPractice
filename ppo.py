# Implementation of PPO
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from torch.distributions import Categorical

# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

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
        self.env = env
        self.action_dim = env.action_space.n
        self.state_dim = env.observation_space.shape[0]

        print(f"action: {self.action_dim}, state: {self.state_dim}")

        self.eps = 0.2
        self.gamma = 0.99
        self.max_ep_length = 1000
        self.samples_per_policy = self.max_ep_length * 4
        self.max_time_steps = 400000
        self.num_epochs = 80
        self.hidden_size = 64
        self.render = True
        self.reward_history = []

        self.data = RolloutBuffer()
        self.critic = Critic(self.state_dim, self.hidden_size).to(device)
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_size).to(device)

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

            episode_reward = 0 # finite time horizons
            for i in range(self.max_ep_length):
                action = self.select_action(state)
                new_state, reward, done, _ = env.step(action)
                time_step += 1
                episode_reward += reward

                if self.render:
                    env.render()
                    # time.sleep(0.01)

                # add data
                self.data.rewards.append(reward)
                self.data.is_terminals.append(done)

                # update if needed
                if time_step % self.samples_per_policy == 0:
                    self.update()

                if done:
                    break

                state = new_state
            print("max episode step reached:", i, " with reward:", episode_reward)
            self.reward_history.append(episode_reward)


    def select_action(self, state):
        # use current policy to sample action
        state = torch.FloatTensor(state).to(device)
        actor_output = self.actor(state)
        dist = Categorical(actor_output)
        action = dist.sample()

        # these data must be added as tensors
        self.data.states.append(state)
        self.data.actions.append(action.detach())
        self.data.log_probs.append(dist.log_prob(action).detach())
        return action.detach().item()

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

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

        rewards = (rewards - rewards.mean()) / rewards.std()

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.data.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.data.actions, dim=0)).detach()
        old_log_probs = torch.squeeze(torch.stack(self.data.log_probs, dim=0)).detach().to(device)

        # evaluate a new policy at the old transitions and take a opt step using gradient of loss
        for i in range(self.num_epochs):
            log_probs, state_values, dist_entropy = self.evaluate(old_states, old_actions)

            ratios = torch.exp(log_probs - old_log_probs)

            # advantages
            state_values = torch.squeeze(state_values).detach()
            advantages = rewards - state_values

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps, 1+self.eps) * advantages

            print(state_values.device(), surr1.device(), surr2.device())

            # if i == 40:
            #     print(torch.min(surr1, surr2).mean().item(), self.MseLoss(rewards, state_values).item(), dist_entropy)
            loss = -1 * torch.min(surr1, surr2) + 0.05 * self.MseLoss(rewards, state_values) - 0.01 * dist_entropy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        self.data.clear()

    # TODO: plot some visualizations to understand training loss
    # avg rewards per episode as t increases (standard plot)
    # action distribution as t increases (bar chart?)
    def plot_visualizations(self):
        fig, axs = plt.subplots(2, 1)
        x_1 = np.arange(len(self.reward_history))
        axs[0].plot(x_1, self.reward_history)
        axs[0].xlabel("episode number")
        axs[0].ylabel("reward")
        axs[0].title("Reward vs. episode number")
        plt.show()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action).to(device)
        state_values = self.critic(state).to(device)
        return action_logprobs, state_values, dist.entropy()

    def unquantize_action(self, action):
        # given action, map to value in (low, high) interval
        delta = self.env.action_space.high - self.env.action_space.low
        return self.env.action_space.low + (delta * action) / self.num_actions

if __name__ == "__main__":
    # for continuous action space, either discretize it or change algo
    env = gym.make("CartPole-v1").env
    ppo = PPO(env)
    ppo.train()
    ppo.plot_visualizations()
