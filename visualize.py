import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

import tkinter as tk
from PIL import Image, ImageTk
from itertools import count, cycle

from celluloid import Camera

# creates video of test run
def visualize(states, arm1_l, arm2_l,dt):

    fig, ax = plt.subplots()
    camera = Camera(fig)
    for state in states:
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        #cart = plt.
        ax.add_patch(Rectangle((state[0]-0.5,-0.125), 1, 0.25))
        arm1_x_endpoint = state[0] + np.sin(state[1]) * arm1_l
        arm1_y_endpoint = np.cos(state[1]) * arm1_l

        ax.plot((state[0], arm1_x_endpoint), (0, arm1_y_endpoint), lw=1.5, c = 'm') 
        arm2_x_endpoint = arm1_x_endpoint+ np.sin(state[2]) * arm2_l
        arm2_y_endpoint = arm1_y_endpoint + np.cos(state[2]) * arm2_l

        ax.plot((arm1_x_endpoint, arm2_x_endpoint), (arm1_y_endpoint, arm2_y_endpoint), lw=1.5, c='r')
        camera.snap()

        
    animation = camera.animate(interval = dt)
    animation.save(f'{get_path_to_saved_images()}/output.gif')
    ax.set_title("Final State")
    


# creates plot of MSE Error over time
def plotter(states, goal, dt):
    states_np = np.array(states)
    goal = np.tile(goal, (states_np.shape[0],1))
    state_mse = np.square(states_np-goal)
    time = np.linspace(0, states_np.shape[0]*dt-dt, states_np.shape[0])

    fig, ax = plt.subplots(3,2)
    fig.tight_layout(h_pad=2)

    ax[0,0].plot(time,state_mse[:,0])
    ax[0,0].title.set_text("cart position MSE vs time")
    ax[1,0].plot(time,state_mse[:,1])
    ax[1,0].title.set_text("theta 1 MSE vs time")
    ax[2,0].plot(time,state_mse[:,2])
    ax[2,0].title.set_text("theta 2 MSE vs time")
    ax[0,1].plot(time,state_mse[:,3])
    ax[0,1].title.set_text("cart velocity MSE vs time")
    ax[1,1].plot(time,state_mse[:,4])
    ax[1,1].title.set_text("theta 1 velocity MSE vs time")
    ax[2,1].plot(time,state_mse[:,5])
    ax[2,1].title.set_text("theta 2 veolicty MSE vs time")
    plt.savefig(f"{get_path_to_saved_images()}/MSE_error.png")
    plt.show()
    plt.close()

# creates plot of MSE Error over time
def plotter_mppi(states, goal, dt):
    states_np = np.array(states)
    goal = np.tile(goal, (states_np.shape[0],1))
    state_mse = np.square(states_np-goal)
    time = np.linspace(0, states_np.shape[0]*dt-dt, states_np.shape[0])

    fig, ax = plt.subplots(3,2)
    fig.tight_layout(h_pad=2)

    ax[0,0].plot(time,state_mse[:,0])
    ax[0,0].title.set_text("cart position MSE vs time")
    ax[1,0].plot(time,state_mse[:,1])
    ax[1,0].title.set_text("theta 1 MSE vs time")
    ax[2,0].plot(time,state_mse[:,2])
    ax[2,0].title.set_text("theta 2 MSE vs time")
    ax[0,1].plot(time,state_mse[:,3])
    ax[0,1].title.set_text("cart velocity MSE vs time")
    ax[1,1].plot(time,state_mse[:,4])
    ax[1,1].title.set_text("theta 1 velocity MSE vs time")
    ax[2,1].plot(time,state_mse[:,5])
    ax[2,1].title.set_text("theta 2 veolicty MSE vs time")
    plt.savefig(f"{get_path_to_saved_images()}/MSE_error.png")
    plt.show()
    plt.close()





def get_path_to_saved_images():
    path_to_images = os.path.realpath(os.path.join(os.path.dirname(__file__), 'saved_data'))

    if not os.path.exists(path_to_images):
       os.mkdir(path_to_images)

    return path_to_images



class ImageLabel(tk.Label):
    """
    A Label that displays images, and plays them if they are gifs
    :im: A PIL Image instance or a string filename
    """
    def load(self, im):
        if isinstance(im, str):
            im = Image.open(im)
        frames = []
 
        try:
            for i in count(1):
                frames.append(ImageTk.PhotoImage(im.copy()))
                im.seek(i)
        except EOFError:
            pass
        self.frames = cycle(frames)
 
        try:
            self.delay = im.info['duration']
        except:
            self.delay = 200
 
        if len(frames) == 1:
            self.config(image=next(self.frames))
        else:
            self.next_frame()
 
    def unload(self):
        self.config(image=None)
        self.frames = None
 
    def next_frame(self):
        if self.frames:
            self.config(image=next(self.frames))
            self.after(self.delay, self.next_frame)


