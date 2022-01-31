import torch
import gym
import time
from numpy import pi
import numpy as np
import random
from collections import defaultdict

# acrobot env is sparsely rewarded env
# goal is to cross the horizontal line above the acrobot in least possible time
# sparse -> random until reward is found

# define constants
num_bins = 9
alpha = .01
gamma = .99
epsilon = .2

env = gym.make("Acrobot-v1")
state = env.reset()

# torch tensor (matrix) of state action pairs
q_func = defaultdict(lambda: np.ones(3))
# q_func = torch.ones((num_bins, num_bins, num_bins, num_bins, num_bins, num_bins, 3))

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))
# print(env.action_space.n, env.observation_space.n)

# given state, put it in 360 different bins (each corresponding to 1 deg)
def quantize_state(state):
    bounds = np.array([1, 1, 1, 1, 4 * pi, 9 * pi])
    max = bounds * 2
    quant = ((state + bounds) * num_bins / max).astype(int)
    return quant

def get_action(state):
    max_q_action = np.argmax(q_func[tuple(state)]) # returns action index 0-2
    if random.random() < epsilon:
        return random.choice([0, 1, 2])
    else:
        return max_q_action
    return 10

def q_update(state, action, reward, new_state, done):
    new_state_max = np.max(q_func[tuple(new_state)])
    q_func[tuple(state)][action] = (1 - alpha) * q_func[tuple(state)][action] + alpha * (reward + gamma * new_state_max)

def reward_shape(reward, state):
    weight = 0.2 # the lower arm (more free one) is 1/5 as important
    return reward - state[0];


num_epochs = 1000
for epoch in range(num_epochs):
    # try q learning by quantizing continuous state space
    state = env.reset()
    done = False
    state = quantize_state(state)
    while not done:
        action = get_action(state)
        new_state, reward, done, info = env.step(action)
        # action corresponds to a sample of the dynamics; let's update
        # modify/engineer rewards to reward y pos of joint
        reward = reward_shape(reward, new_state)
        new_state = quantize_state(new_state)
        q_update(state, action, reward, new_state, done) # sarsd tuple

        state = new_state
        if (epoch + 1) % 100 == 0:
            print(epoch + 1, " Reward:", reward)
            env.render()
        # time.sleep(.03)

# quantize_state(state)
