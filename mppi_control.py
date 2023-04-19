import torch
from torch.distributions.multivariate_normal import MultivariateNormal


def get_cartpole_mppi_hyperparams():
    """
    Returns a dictionary containing the hyperparameters for running MPPI on the cartpole environment
    The required parameters are:
     * lambda: float parameter between 0. and 1. used to weight samples.
     * Q: torch tensor fo shape (state_size, state_size) representing the state quadratic cost.
     * noise_sigma: torch tensor fo size (action_size, action_size) representing the covariance matrix  of the random action perturbations.
    """
    action_size = 1
    state_size = 4
    hyperparams = {
        'lambda': None,
        'Q': None,
        'noise_sigma': None,
    }
    # --- Your code here

    hyperparams['lambda'] = 0.05
    x = 45.0 
    theta = 100.0
    x_dot = 15.0 
    theta_dot = 1.0 
    hyperparams['Q'] = torch.diag(torch.tensor([x, theta, x_dot, theta_dot]))
    hyperparams['noise_sigma'] = torch.eye(action_size) * 25

    # ---
    return hyperparams


def get_panda_mppi_hyperparams():
    """
    Returns a dictionary containing the hyperparameters for running MPPI on the panda environment
    The required parameters are:
     * lambda: float parameter between 0. and 1. used to weight samples.
     * Q: torch tensor fo shape (state_size, state_size) representing the state quadratic cost.
     * noise_sigma: torch tensor fo size (action_size, action_size) representing the covariance matrix  of the random action perturbations.
    """
    action_size = 7
    state_size = 14
    hyperparams = {
        'lambda': None,
        'Q': None,
        'noise_sigma': None,
    }
    # --- Your code here

    hyperparams['lambda'] = 0.01
    x1 = 5000.0
    x2 = 5000.0
    x3 = 5000.0
    x4 = 5000.0
    x5 = 5000.0
    x6 = 5000.0
    x7 = 5000.0

    x1_dot = 50.0
    x2_dot = 50.0
    x3_dot = 50.0
    x4_dot = 50.0
    x5_dot = 50.0
    x6_dot = 50.0
    x7_dot = 50.0
    

    hyperparams['Q'] = torch.diag(torch.tensor([x1, x2, x3, x4, x5, x6, x7, x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot, x7_dot]))
    hyperparams['noise_sigma'] = torch.eye(action_size) * 125
    # ---
    return hyperparams


class MPPIController(object):

    def __init__(self, env, num_samples, horizon, hyperparams):
        """

        :param env: Simulation environment. Must have an action_space and a state_space.
        :param num_samples: <int> Number of perturbed trajectories to sample
        :param horizon: <int> Number of control steps into the future
        :param hyperparams: <dic> containing the MPPI hyperparameters
        """
        self.env = env
        self.T = horizon
        self.K = num_samples
        self.lambda_ = hyperparams['lambda']
        self.action_size = env.action_space.shape[-1]
        self.state_size = env.state_space.shape[-1]
        self.goal_state = torch.zeros(self.state_size)  # This is just a container for later use
        self.Q = hyperparams['Q'] # Quadratic Cost Matrix (state_size, state_size)
        self.noise_mu = torch.zeros(self.action_size)
        self.noise_sigma = hyperparams['noise_sigma']  # Noise Covariance matrix shape (action_size, action_size)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.U = torch.zeros((self.T, self.action_size)) # nominal action sequence (T, action_size)
        self.u_init = torch.zeros(self.action_size)
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)

    def reset(self):
        """
        Resets the nominal action sequence
        :return:
        """
        self.U = torch.zeros((self.T, self.action_size))# nominal action sequence (T, action_size)

    def command(self, state):
        """
        Run a MPPI step and return the optimal action.
        :param state: torch tensor of shape (state_size,)
        :return:
        """
        action = None
        perturbations = self.noise_dist.sample((self.K, self.T))    # shape (K, T, action_size)
        perturbed_actions = self.U + perturbations      # shape (K, T, action_size)
        trajectory = self._rollout_dynamics(state, actions=perturbed_actions)
        trajectory_cost = self._compute_trajectory_cost(trajectory, perturbations)
        self._nominal_trajectory_update(trajectory_cost, perturbations)
        # select optimal action
        action = self.U[0]
        # final update nominal trajectory
        self.U = torch.roll(self.U, -1, dims=0) # move u_t to u_{t-1}
        self.U[-1] = self.u_init # Initialize new end action
        return action

    def _rollout_dynamics(self, state_0, actions):
        """
        Roll out the environment dynamics from state_0 and taking the control actions given by actions
        :param state_0: torch tensor of shape (state_size,)
        :param actions: torch tensor of shape (K, T, action_size)
        :return:
         * trajectory: torch tensor of shape (K, T, state_size) containing the states along the trajectories given by
                       starting at state_0 and taking actions.
                       This tensor contains K trajectories of T length.
         TIP 1: You may need to call the self._dynamics method.
         TIP 2: At most you need only 1 for loop.
        """
        state = state_0.unsqueeze(0).repeat(self.K, 1) # transform it to (K, state_size)
        trajectory = None
        # --- Your code here

        state_size = state_0.size(dim=0)
        K = actions.size(dim=0)
        T = actions.size(dim=1)
        
        #state_0_repeated = state_0.repeat(K,1)
        trajectory = torch.zeros((K,T, state_size))

        #print(self._dynamics(state, actions[:,0,:]).shape)
        trajectory[:,0,:] = self._dynamics(state, actions[:,0,:])

        for t in range(1,T,1):
          #print(actions[:,t,:].size())
          #print(trajectory)
          trajectory[:,t,:] = self._dynamics(trajectory[:,t-1,:], actions[:,t,:])

        # ---
        return trajectory

    def _compute_trajectory_cost(self, trajectory, perturbations):
        """
        Compute the costs for the K different trajectories
        :param trajectory: torch tensor of shape (K, T, state_size)
        :param perturbations: torch tensor of shape (K, T, action_size)
        :return:
         - total_trajectory_cost: torch tensor of shape (K,) containing the total trajectory costs for the K trajectories
        Observations:
        * The trajectory cost be the sum of the state costs and action costs along the trajectories
        * State cost should be quadratic as (state_i-goal_state)^T Q (state_i-goal_state)
        * Action costs should be given by (non_perturbed_action_i)^T noise_sigma^{-1} (perturbation_i)

        TIP 1: the nominal actions (without perturbation) are stored in self.U
        TIP 2: Check Algorithm 2 in https://ieeexplore.ieee.org/document/7989202 for more references.
        """
        total_trajectory_cost = None
        # --- Your code here

        
        K = trajectory.size(dim=0)
        T = trajectory.size(dim=1)

        goal_expanded = self.goal_state.repeat(K,T,1)
        #print(self.noise_sigma_inv.shape)
        #print(perturbations.transpose(2,1).shape)

        total_trajectory_cost = torch.diagonal((trajectory - goal_expanded) @ self.Q @ torch.transpose(trajectory - goal_expanded, 2,1), dim1 = 1, dim2 = 2).sum(dim=1) \
          + torch.diagonal(self.lambda_ *  (self.U @ self.noise_sigma_inv @ perturbations.transpose(2,1)), dim1=1, dim2=2).sum(dim=1)
        #print("trajectory cost")
        #print(total_trajectory_cost.shape)
        #print(total_trajectory_cost)

        # ---
        return total_trajectory_cost

    def _nominal_trajectory_update(self, trajectory_costs, perturbations):
        """
        Update the nominal action sequence (self.U) given the trajectory costs and perturbations
        :param trajectory_costs: torch tensor of shape (K,)
        :param perturbations: torch tensor of shape (K, T, action_size)
        :return: No return, you just need to update self.U

        TIP: Check Algorithm 2 in https://ieeexplore.ieee.org/document/7989202 for more references about the action update.
        """
        # --- Your code here

        K = perturbations.size(dim=0)
        T = perturbations.size(dim=1)
        action_size = perturbations.size(dim=2)


        #print(trajectory_costs.shape)
        min_cost = torch.min(trajectory_costs)
        normalization_values = torch.exp(-1/self.lambda_ * (trajectory_costs - min_cost))
        #print(normalization_values.shape)
        normalization = torch.sum(normalization_values)
        weights = (1/normalization) * normalization_values
        #print(weights.shape)
        weights = weights.reshape(-1,1)
        #print(weights.shape)
        weights_expanded = weights.repeat(1,T).reshape(K,T,1)

        #print(weights_expanded.shape)
        weights_expanded = weights_expanded.repeat(1,1,action_size)
        #print(weights_expanded.shape)
        #print(perturbations.shape)

        actions = weights_expanded * perturbations
        self.U = torch.sum(actions, dim=0)

        # ---

    def _dynamics(self, state, action):
        """
        Query the environment dynamics to obtain the next_state in a batched format.
        :param state: torch tensor of size (...., state_size)
        :param action: torch tensor of size (..., action_size)
        :return: next_state: torch tensor of size (..., state_size)
        """
        next_state = self.env.batched_dynamics(state.cpu().detach().numpy(), action.cpu().detach().numpy())
        next_state = torch.tensor(next_state, dtype=state.dtype)
        return next_state

