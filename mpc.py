import torch

from equations_motion import motionModel
from visualize import visualize, plotter
from cost import Cost
from iLQR import iLQR
from dynamics import Dynamics

def wrapToPI(phase):
    x_wrap = phase% (2 * torch.pi)
    while abs(x_wrap) > torch.pi:
        x_wrap -= 2 * torch.pi * (x_wrap/abs(x_wrap))
    return x_wrap



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
num_steps = 25
controller = iLQR(double_pend_dynamics, cost_fn, initial_state, goal_state, num_steps=num_steps)
system_states = []
system_states.append(initial_state)

U = 20*torch.rand((num_steps, 1))-10   #initialize random actions
for i in range(150):
    print(i)
    action = 0
    current_state = system_states[-1]
    U = controller.ilqr(current_state, U)
    next_state = motion_model.f(current_state, U[0,:])
    U_next = torch.zeros_like(U)
    U_next[0:num_steps-1,:] = U[1:num_steps,:]
    U = U_next
    system_states.append(next_state)
print(U)
system_states_np = []
for i in range(len(system_states)):
    state_np = system_states[i].detach().numpy()
    state_np[1] = wrapToPI(state_np[1])
    state_np[2] = wrapToPI(state_np[2])
    system_states_np.append(state_np)


visualize(system_states_np, length_arm1, length_arm2, dt)

plotter(system_states_np, goal_state.numpy(), dt)
