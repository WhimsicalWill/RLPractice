from dm_control import suite
from PIL import Image
import subprocess
import numpy as np

domain = "cheetah"
task = "run"
env = suite.load(domain_name=domain, task_name=task)

action_spec = env.action_spec()
time_step = env.reset()

counter = 0

subprocess.call(['rm', '-rf', 'frames'])
subprocess.call(['mkdir', '-p', 'frames'])

# print(action_spec.minimum, action_spec.maximum)
constant_action = np.array([.5, .5, .5, .5, .5, .5])

while not time_step.last():
	# action = np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)
	time_step = env.step(constant_action)
	print(time_step.reward, time_step.discount, time_step.observation)
	
	# save image to video
	image_data = np.hstack([env.physics.render(height=400, width=400, camera_id=0),
	env.physics.render(height=400, width=400, camera_id=1)])
	
	
	# hv stack two different camera views to an image 
	img = Image.fromarray(image_data, 'RGB')
	img.save(f"frames/frame-%.10d.png" % counter)
	counter += 1
	
subprocess.call(['ffmpeg', '-framerate', '50', '-y', '-i', 'frames/frame-%010d.png', '-r',
'30', '-pix_fmt', 'yuv420p', 'video.mp4'])
