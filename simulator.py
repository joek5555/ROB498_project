import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from numpngw import write_apng
import imageio
import os

arm1_l = 1.0
arm2_l = 1.0
dt = 0.1



def get_path_to_saved_images():
    path_to_images = os.path.realpath(os.path.join(os.path.dirname(__file__), 'saved_images'))

    if not os.path.exists(path_to_images):
       os.mkdir(path_to_images)

    return path_to_images


def visualize(states):
    i = 0.0
    frames = []
    for state in states:
        fig, ax = plt.subplots()
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        #cart = plt.
        print(state)
        ax.add_patch(Rectangle((state[0]-0.5,-0.125), 1, 0.25))
        arm1_x_endpoint = state[0] + np.sin(state[1]) * arm1_l
        arm1_y_endpoint = np.cos(state[1]) * arm1_l

        ax.plot((state[0], arm1_x_endpoint), (0, arm1_y_endpoint), lw=1.5, c = 'm') 
        arm2_x_endpoint = arm1_x_endpoint+ np.sin(state[2]) * arm2_l
        arm2_y_endpoint = arm1_y_endpoint + np.cos(state[2]) * arm2_l

        ax.plot((arm1_x_endpoint, arm2_x_endpoint), (arm1_y_endpoint, arm2_y_endpoint), lw=1.5, c='r')

        image_path = get_path_to_saved_images()
        plt.savefig(f'{image_path}/state_t_{round(i,1)}.png')
        frames.append(imageio.imread(f'{image_path}/state_t_{round(i,1)}.png'))
        i += dt
        plt.close()

    
    imageio.mimsave('animation.gif', frames, fps=dt)

states = np.vstack([np.arange(0,6,0.2), np.arange(0,3,0.1), np.arange(0,6,0.2)]).T
state = states.tolist()

#state = [np.array([0,0,0]), np.array([0.1,0,0]), np.array([0.2,0,0]), np.array([0.3,0,0]), np.array([0.4,0,0])]
visualize(state)

