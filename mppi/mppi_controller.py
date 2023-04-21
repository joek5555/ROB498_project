import torch
from mppi import MPPI


class SwingupController(object):

    def __init__(self, motion_model, cost_function,state_dim = 6, action_dim = 1, num_samples = 100, horizon=30, u_min = -1750, u_max= 1750):

        self.model = motion_model
        self.target_state = None
        noise_sigma = 25 * torch.eye(action_dim)
        lambda_value = 0.01

        
        self.mppi = MPPI(self._compute_dynamics,
                         cost_function,
                         nx=state_dim,
                         num_samples=num_samples,
                         horizon=horizon,
                         noise_sigma=noise_sigma,
                         lambda_=lambda_value,
                         u_min=u_min,
                         u_max=u_max)
        
    def _compute_dynamics(self, state, action):
        """
        Compute next_state using the dynamics model self.model and the provided state and action tensors
        :param state: torch tensor of shape (B, state_size)
        :param action: torch tensor of shape (B, action_size)
        :return: next_state: torch tensor of shape (B, state_size) containing the predicted states from the learned model.
        """
        next_state = None
        #print(state[0,:].squeeze())

        # --- Your code here
        next_state = torch.zeros_like(state)
        for i in range(state.shape[0]):
        
            next_state[i,:] = self.model(state[i,:].squeeze(),action[i])


        # ---
        return next_state
    
    
    def control(self, state):
        """
        Query MPPI and return the optimal action given the current state <state>
        :param state: numpy array of shape (state_size,) representing current state
        :return: action: numpy array of shape (action_size,) representing optimal action to be sent to the robot.
        TO DO:
         - Prepare the state so it can be send to the mppi controller. Note that MPPI works with torch tensors.
         - Unpack the mppi returned action to the desired format.
        """
        action = None
        state_tensor = None
        # --- Your code here


        # ---
        action_tensor = self.mppi.command(state)
        # --- Your code here
        
        action = action_tensor.detach()

        # ---
        return action