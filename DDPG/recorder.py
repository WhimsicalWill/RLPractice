from PIL import Image
import subprocess
import numpy as np

class Recorder():
    def __init__(self, env):
        self.counter = 0
        self.env = env

    def clear_directory(self):
        subprocess.call(['rm', '-rf', 'frames'])
        subprocess.call(['mkdir', '-p', 'frames'])
    
    def save_frame(self):
        image_data = np.hstack([self.env.physics.render(height=400, width=400, camera_id=0),
	    self.env.physics.render(height=400, width=400, camera_id=1)])
        # hv stack two different camera views to an image 
        img = Image.fromarray(image_data, 'RGB')
        img.save(f"./frames/frame-%.10d.png" % self.counter)
        self.counter += 1

    # render a video from the images in the frames dir
    def render_video(self):
        subprocess.call(['ffmpeg', '-framerate', '50', '-y', '-i', 'frames/frame-%010d.png', '-r',
'30', '-pix_fmt', 'yuv420p', 'video.mp4'])