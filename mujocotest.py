import gym
import mujoco_py
from gym.wrappers import Monitor

env = gym.make('Reacher-v2')
env = Monitor(env, 'recording', video_callable=lambda episode_id: episode_id==1, force=True)

max_episodes = 100
max_time_steps = 300

for i in range(max_episodes):
	env.reset()
	for t in range(max_time_steps):
		env.render()
		new_state, reward, done, _ = env.step(env.action_space.sample())
		
		if done:
			break
env.close()
