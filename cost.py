import torch
class Cost():
    def __init__(self):
        self.x_dim = 6
        self.u_dim = 1
        self.Q = torch.eye(self.x_dim)
        self.Q[0,0] = 1.2
        self.Q[1,1] = 2
        self.Q[2,2] = 2
        self.Q[3,3] = 1.4
        self.Q[4,4] = 1.7
        self.Q[5,5] = 1.7
        self.R = torch.eye(self.u_dim)

    def l(self, state, goal_state, action=None):
        state = state.unsqueeze(1)
        goal_state = goal_state.unsqueeze(1)
        if (action == None):
            return torch.t(state-goal_state) @ self.Q @ (state - goal_state)
        cost = torch.t(state-goal_state) @ self.Q @ (state - goal_state) + torch.t(action) @ self.R @ action
        return cost
    
    def l_x(self, x, x_goal, u=None):    #make sure to pass in x and u as tensor with requires_grad = True
        l = self.l(x, x_goal, u)
        grad = []
        for i in range(l.shape[0]):
            dxi = torch.autograd.grad(l[i], x, allow_unused=True, create_graph=True)[0]
            grad.append(dxi)
        grad = torch.stack(grad)
        return grad
    
    def l_u(self, x, x_goal, u=None):    #make sure to pass in x and u as tensor with requires_grad = True
        l = self.l(x, x_goal, u)
        grad = []
        for i in range(l.shape[0]):
            dui = torch.autograd.grad(l[i], u, allow_unused=True, create_graph=True)[0]
            grad.append(dui)
        grad = torch.stack(grad)
        return grad      #returns tensor
    
    def l_xx(self, x, x_goal, u=None):
        l_x = self.l_x(x, x_goal, u)
        l_xx = []
        for i in range(l_x.shape[0]):
            row = []
            for j in range(l_x.shape[1]):
                hessian = torch.autograd.grad(l_x[i][j], x, allow_unused=True, create_graph=True)[0]
                if (hessian == None):
                    hessian = torch.zeros((1))
                row.append(hessian)
            row = torch.stack(row)
            l_xx.append(row)
        l_xx = torch.stack(l_xx)
        return l_xx
    
    def l_xu(self, x, x_goal, u=None):
        l_x = self.l_x(x, x_goal, u)
        l_xu = []
        for i in range(l_x.shape[0]):
            row = []
            for j in range(l_x.shape[1]):
                hessian = torch.autograd.grad(l_x[i][j], u, allow_unused=True, create_graph=True)[0]
                if (hessian == None):
                    hessian = torch.zeros((1))
                row.append(hessian)
            row = torch.stack(row)
            l_xu.append(row)
        l_xu = torch.stack(l_xu)
        return l_xu
    
    def l_ux(self, x, x_goal, u=None):
        l_u = self.l_u(x, x_goal, u)
        l_ux = []
        for i in range(l_u.shape[0]):
            row = []
            for j in range(l_u.shape[1]):
                hessian = torch.autograd.grad(l_u[i][j], x, allow_unused=True, create_graph=True)[0]
                if (hessian == None):
                    hessian = torch.zeros((6))
                row.append(hessian)
            row = torch.stack(row)
            l_ux.append(row)
        l_ux = torch.stack(l_ux)
        return l_ux
    
    def l_uu(self, x, x_goal, u=None):
        if (u == None):
            return torch.zeros((1)).unsqueeze(0).unsqueeze(0)
        l_u = self.l_u(x, x_goal, u)
        l_uu = []
        for i in range(l_u.shape[0]):
            row = []
            for j in range(l_u.shape[1]):
                hessian = torch.autograd.grad(l_u[i][j], u, allow_unused=True, create_graph=True)[0]
                if (hessian == None):
                    hessian = torch.zeros((1))
                row.append(hessian)
            row = torch.stack(row)
            l_uu.append(row)
        l_uu = torch.stack(l_uu)
        return l_uu