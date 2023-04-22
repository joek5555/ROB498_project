import torch
import argparse
from equations_motion import motionModel
from visualize import visualize, plotter, plotter_mppi
from cost import Cost
from mpc import MPC
from mppi_controller import SwingupController
from dynamics import Dynamics

def wrapToPI(phase):
    x_wrap = phase% (2 * torch.pi)
    while abs(x_wrap) > torch.pi:
        x_wrap -= 2 * torch.pi * (x_wrap/abs(x_wrap))
    return x_wrap



def setController(c):

    if c != "MPPI" and c != "iLQR" and c != "DDP":
        raise argparse.ArgumentTypeError("controller type can only be MPPI, DDP, or iLQR")
    return c


parser = argparse.ArgumentParser(description='Run localization')
parser.add_argument('-c', '--controller_type_input', type=setController, help= 'set controller type. Valid options are MPPI, DDP, or iLQR')
args = parser.parse_args()


controller_type = args.controller_type_input
if controller_type is None:
    controller_type = "DDP"
print(f' using {controller_type} controller.')
print('Note that you can specify from the command line what controller to use using the -c flag')
print('Valid types are MPPI, DDP, or iLQR.')

dt = 0.05
mass_arm1 = 0.5 
mass_arm2 = 0.5 
mass_cart = 1.0 
length_arm1 = 1.0
length_arm2 = 1.0

motion_model = motionModel(mass_cart, mass_arm1, mass_arm2, length_arm1, length_arm2)

initial_state = torch.tensor([0.0, 3.14, 3.14, 0.0, 0.0, 0.0], requires_grad=True)
goal_state = torch.zeros((6))
double_pend_dynamics = Dynamics(x_dim=6, u_dim=1, motion_model=motion_model)
cost_fn = Cost()




if controller_type == "iLQR" or controller_type == "DDP":
    num_steps = 150
    horizon = 25
    controller = MPC(double_pend_dynamics, cost_fn, initial_state, goal_state, horizon=horizon, controller_type =controller_type)
    system_states = []
    system_states.append(initial_state)

    U = 20*torch.rand((horizon, 1))-10   #initialize random actions

    i = 0
    while(i < num_steps):
        print(f'-------iteration {i}----------')
        action = 0
        current_state = system_states[-1]
        U = controller.mpc(current_state, U)
        if U == "ERROR":
            i =0
            U = 20*torch.rand((horizon, 1))-10   #initialize random actions
            continue
        next_state = motion_model.f(current_state, U[0,:])
        U_next = torch.zeros_like(U)
        U_next[0:horizon-1,:] = U[1:horizon,:]
        U = U_next
        system_states.append(next_state)

        i+=1

    system_states_np = []
    for i in range(len(system_states)):
        state_np = system_states[i].detach().numpy()
        state_np[1] = wrapToPI(state_np[1])
        state_np[2] = wrapToPI(state_np[2])
        system_states_np.append(state_np)

    visualize(system_states_np, length_arm1, length_arm2, dt)

    plotter(system_states_np, goal_state.numpy(), dt)



elif controller_type == "MPPI":
    num_steps = 100
    horizon = 35
    x_dim = 6
    u_dim = 1
    noise_sigma = torch.tensor([0.5])
    lambda_value = 0.01
    u_min = torch.tensor([-75])
    u_max = torch.tensor([75])
    mppi_controller = SwingupController(motion_model.f, cost_fn.cost_batch_mppi, x_dim, u_dim,num_steps, horizon,u_min = u_min, u_max = u_max )

    initial_state = torch.tensor([0, 3.14, 3.14, 0, 0, 0])
    system_states = []
    system_states.append(initial_state)

    for i in range(num_steps):
        print(f'-------iteration {i}----------')
        action = mppi_controller.control(system_states[-1])
        next_state = motion_model.f(system_states[-1], action)
        #next_state = true_motion_model.f(system_states[-1], action)
        system_states.append(next_state)

        
    system_states_np = []
    for i in range(len(system_states)):

        system_states_np.append(system_states[i].detach().numpy())



    visualize(system_states_np, length_arm1, length_arm2, dt)

    plotter_mppi(system_states_np, goal_state.numpy(), dt)
