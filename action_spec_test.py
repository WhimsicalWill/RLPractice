from dm_control import suite
from PIL import Image
import subprocess
import numpy as np

domain = "cheetah"
task = "run"
env = suite.load(domain_name=domain, task_name=task)

action_spec = env.action_spec()
time_step = env.reset()

print(action_spec)
print(time_step)

print("<---------->")

print(time_step.observation['position'])
print(time_step.observation['velocity'])

print(len(time_step.observation['position']))
print(len(time_step.observation['velocity']))
