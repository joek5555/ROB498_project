import torch

from equations_motion import motionModel
from visualize import visualize
from cost import Cost
from iLQR import iLQR
from dynamics import Dynamics

dt = 0.05
runtime = 10 # in seconds

mass_arm1 = 84.01 # (l=1, r=0.1) * 2710 
mass_arm2 = 84.01 # (l=1, r=0.1) * 2710
mass_cart = 169.375 # (1 * 0.25 * 0.25 ) * 2710
length_arm1 = 1.0
length_arm2 = 1.0

motion_model = motionModel(mass_cart, mass_arm1, mass_arm2, length_arm1, length_arm2)

initial_state = torch.tensor([0, 3.14, 3.14, 0, 0, 0], requires_grad=True)
goal_state = torch.zeros((6))
double_pend_dynamics = Dynamics(x_dim=6, u_dim=1, motion_model=motion_model)
cost_fn = Cost()
controller = iLQR(double_pend_dynamics, cost_fn, initial_state, goal_state)
U = controller.ilqr()

system_states = []
system_states.append(initial_state)

for i in range(U.shape[0]):
    action = 0
    current_state = system_states[-1]

    next_state = motion_model.f(current_state, U[i,:])
    #print(next_state)
    system_states.append(next_state)

system_states_np = []
for i in range(len(system_states)):
    system_states_np.append(system_states[i].detach().numpy())


visualize(system_states_np, dt)
