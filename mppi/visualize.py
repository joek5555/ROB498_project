import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

from celluloid import Camera

arm1_l = 1.0
arm2_l = 1.0

"""
def get_path_to_saved_images():
    path_to_images = os.path.realpath(os.path.join(os.path.dirname(__file__), 'saved_images'))

    if not os.path.exists(path_to_images):
       os.mkdir(path_to_images)

    return path_to_images
"""

def visualize(states,dt):
    i = 0.0
    frames = []
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
        #image_path = get_path_to_saved_images()
        #plt.savefig(f'{image_path}/state_t_{round(i,1)}.png')
        #frames.append(imageio.imread(f'{image_path}/state_t_{round(i,1)}.png'))
        i += dt
        
    animation = camera.animate(interval = dt)
    animation.save('output.gif')
    
    #imageio.mimsave('animation.gif', frames, fps=dt)

#states = np.vstack([np.arange(0,6,0.2), np.arange(0,3,0.1), np.arange(0,6,0.2)]).T
#state = states.tolist()

#state = [np.array([0,0,0]), np.array([0.1,0,0]), np.array([0.2,0,0]), np.array([0.3,0,0]), np.array([0.4,0,0])]
#visualize(state, dt=0.1)

def plotter(states, dt, label = 0):
    states_np = np.array(states)
    fig, ax = plt.subplots(3,2)
    fig.tight_layout(h_pad=2)
    time = np.arange(0, states_np.shape[0]*dt-dt, dt)
    ax[0,0].plot(time,states_np[:,0])
    ax[0,0].title.set_text("cart position vs time")
    ax[1,0].plot(time,states_np[:,1])
    ax[1,0].title.set_text("theta 1 vs time")
    ax[2,0].plot(time,states_np[:,2])
    ax[2,0].title.set_text("theta 2 vs time")
    ax[0,1].plot(time,states_np[:,3])
    ax[0,1].title.set_text("cart velocity vs time")
    ax[1,1].plot(time,states_np[:,4])
    ax[1,1].title.set_text("theta 1 velocity vs time")
    ax[2,1].plot(time,states_np[:,5])
    ax[2,1].title.set_text("theta 2 veolicty vs time")
    plt.savefig(f"{get_path_to_saved_images()}/state_{label}.png")
    plt.close()






def get_path_to_saved_images():
    path_to_images = os.path.realpath(os.path.join(os.path.dirname(__file__), 'saved_images'))

    if not os.path.exists(path_to_images):
       os.mkdir(path_to_images)

    return path_to_images

