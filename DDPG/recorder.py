from PIL import Image
import subprocess
import numpy as np
import matplotlib.pyplot as plt

class Recorder():
    def __init__(self, env):
        self.frame_count = 0
        self.ep_count = 0
        self.env = env

    def clear_directory(self):
        subprocess.call(['rm', '-rf', 'frames'])
        subprocess.call(['mkdir', '-p', 'frames'])
        subprocess.call(['rm', '-rf', 'rewards'])
        subprocess.call(['mkdir', '-p', 'rewards'])
    
    def save_frame(self):
        image_data = np.hstack([self.env.physics.render(height=400, width=400, camera_id=0),
	    self.env.physics.render(height=400, width=400, camera_id=1)])
        # hv stack two different camera views to an image 
        img = Image.fromarray(image_data, 'RGB')
        img.save("./frames/frame-%.10d.png" % self.frame_count)
        self.counter += 1

    def save_rewards(self, rewards):
        # create matplotlib diagram for reward in epoch and save
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.arange(len(rewards)), rewards)
        plt.savefig("./rewards/rewards%.10d.png" % self.ep_count)
        plt.close()
        self.ep_count += 1

        

    # render a video from the images in the frames dir
    def render_video(self):
        subprocess.call(['ffmpeg', '-framerate', '50', '-y', '-i', 'frames/frame-%010d.png', '-r',
'30', '-pix_fmt', 'yuv420p', 'video.mp4'])