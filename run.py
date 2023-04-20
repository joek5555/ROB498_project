import torch

from equations_motion import motionModel
from visualize import visualize

dt = 0.05
runtime = 10 # in seconds

mass_arm1 = 84.01 # (l=1, r=0.1) * 2710 
mass_arm2 = 84.01 # (l=1, r=0.1) * 2710
mass_cart = 169.375 # (1 * 0.25 * 0.25 ) * 2710
length_arm1 = 1.0
length_arm2 = 1.0

motion_model = motionModel(mass_cart, mass_arm1, mass_arm2, length_arm1, length_arm2)

system_states = []
initial_state = torch.tensor([0, -0.7854, 0.7854, 0, 0, 0])
system_states.append(initial_state)

for t in range(0,int(10/dt)):
    action = 0
    current_state = system_states[-1]

    next_state = motion_model.motion(current_state, action)
    #print(next_state)
    system_states.append(next_state)

visualize(system_states, dt)
