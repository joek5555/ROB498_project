import torch
class Cost():
    def __init__(self):
        self.x_dim = 6
        self.u_dim = 1
        self.Q = torch.eye(self.x_dim)
        self.Q[0,0] = 45.0
        self.Q[1,1] = 100.0
        self.Q[2,2] = 100.0
        self.Q[3,3] = 2.0
        self.Q[4,4] = 6.0
        self.Q[5,5] = 0.01
        self.Q_f = torch.eye(self.x_dim)
        self.Q_f[0,0] = 1.0
        self.Q_f[1,1] = 20.0
        self.Q_f[2,2] = 20.0
        self.Q_f[3,3] = 0.01
        self.Q_f[4,4] = 0.05
        self.Q_f[5,5] = 0.05
        self.R = torch.eye(self.u_dim)
        self.R[0,0] = 0.001

    def cost(self, state, goal_state, action=None):
        state = state.unsqueeze(1)
        # with torch.no_grad():
        #     state[1] = self.wrapToPI(state[1])
        #     state[2] = self.wrapToPI(state[2])
        goal_state = goal_state.unsqueeze(1)
        if (action == None):
            return torch.t(state-goal_state) @ self.Q_f @ (state - goal_state)
        cost = torch.t(state-goal_state) @ self.Q @ (state - goal_state) + torch.t(action) @ self.R @ action
        return cost
    
    def total_cost(self, X, goal_state, U):
        cost = 0
        for i in range(U.shape[0]):
            cost += self.l(X[i,:], goal_state, U[i,:])
        cost += self.l(X[-1,:], goal_state)
        return cost
    
    def cost_batch(self, state, action):
        goal_state = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        state_diff = state - torch.tile(goal_state.reshape(1,-1), (state.shape[0], 1))
        cost = torch.diagonal(state_diff @ self.Q @ state_diff.T) + (torch.square(action) * self.R[0,0]).squeeze() 
        return cost