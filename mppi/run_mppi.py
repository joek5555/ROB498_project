import torch
import os
import sys
from visualize import visualize

#path_to_files = os.path.realpath(os.path.join(os.path.dirname(__file__), 'mppi'))

#sys.path.insert(1, path_to_files)
from mppi_controller import SwingupController
from equations_motion_mppi import motionModel
from cost_mppi import Cost


dt = 0.05

mass_arm1 = 0.5 # (l=1, r=0.1) * 2710 
mass_arm2 = 0.5 # (l=1, r=0.1) * 2710
mass_cart = 1.0 # (1 * 0.25 * 0.25 ) * 2710
length_arm1 = 1.0
length_arm2 = 1.0

motion_model = motionModel(mass_cart, mass_arm1, mass_arm2, length_arm1, length_arm2)
cost_class = Cost()

num_steps = 100
horizon = 40
x_dim = 6
u_dim = 1
noise_sigma = torch.tensor([0.5])
lambda_value = 0.01
u_min = torch.tensor([-75])
u_max = torch.tensor([75])
mppi_controller = SwingupController(motion_model.f, cost_class.cost_batch, x_dim, u_dim,num_steps, horizon,u_min = u_min, u_max = u_max )

initial_state = torch.tensor([0, 3.14, 3.14, 0, 0, 0])
system_states = []
system_states.append(initial_state)

for i in range(num_steps):
    print(f'-------iteration {i}----------')
    action = mppi_controller.control(system_states[-1])
    next_state = motion_model.f(system_states[-1], action)
    system_states.append(next_state)
    print(next_state)
    
system_states_np = []
for i in range(len(system_states)):

    system_states_np.append(system_states[i].detach().numpy())



visualize(system_states_np, dt)