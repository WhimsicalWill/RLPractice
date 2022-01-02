import torch
import gym
import time
from numpy import pi
import numpy as np
import random

env = gym.make("CartPole-v1")
state = env.reset()

for epoch in range(10):
    for i in range(100):
        next_state, reward, done, info = env.step(env.action_space.sample())
        if done:
            break
