import torch

class Dynamics():
    def __init__(self, x_dim, u_dim, motion_model):
        self.x_dim = x_dim  # state dimension
        self.u_dim = u_dim  # control input dimension
        self.motion_model = motion_model
    
    def f(self, x, u):      #our dynamics
        return self.motion_model.f(x,u)  #return x_{t+1}
    
    def rollout(self, x_0, U):
        X = x_0.unsqueeze(0)
        for i in range(U.shape[0]):
            x_i = self.f(X[i,:], U[i,:])
            X = torch.cat((X, x_i.unsqueeze(0)), dim = 0)
        return X
    
    def f_x(self, x, u):    #make sure to pass in x and u as tensor with requires_grad = True
        f = self.f(x, u)
        grad = []
        for i in range(f.shape[0]):
            dxi = torch.autograd.grad(f[i], x, allow_unused=True, create_graph=True)[0]
            grad.append(dxi)
        grad = torch.stack(grad)
        return grad
    
    def f_u(self, x, u):    #make sure to pass in x and u as tensor with requires_grad = True
        f = self.f(x, u)
        grad = []
        for i in range(f.shape[0]):
            dui = torch.autograd.grad(f[i], u, allow_unused=True, create_graph=True)[0]
            grad.append(dui)
        grad = torch.stack(grad)
        return grad      #returns tensor
    
    def f_xx(self, x, u):
        f_x = self.f_x(x, u)
        f_xx = []
        for i in range(f_x.shape[0]):
            row = []
            for j in range(f_x.shape[1]):
                hessian = torch.autograd.grad(f_x[i][j], x, allow_unused=True, create_graph=True)[0]
                if (hessian == None):
                    hessian = torch.zeros((1))
                row.append(hessian)
            row = torch.stack(row)
            f_xx.append(row)
        f_xx = torch.stack(f_xx)
        return f_xx
    
    def f_xu(self, x, u):
        f_x = self.f_x(x, u)
        f_xu = []
        for i in range(f_x.shape[0]):
            row = []
            for j in range(f_x.shape[1]):
                hessian = torch.autograd.grad(f_x[i][j], u, allow_unused=True, create_graph=True)[0]
                if (hessian == None):
                    hessian = torch.zeros((1))
                row.append(hessian)
            row = torch.stack(row)
            f_xu.append(row)
        f_xu = torch.stack(f_xu)
        return f_xu
    
    def f_ux(self, x, u):
        f_u = self.f_u(x, u)
        f_ux = []
        for i in range(f_u.shape[0]):
            row = []
            for j in range(f_u.shape[1]):
                hessian = torch.autograd.grad(f_u[i][j], x, allow_unused=True, create_graph=True)[0]
                if (hessian == None):
                    hessian = torch.zeros((1))
                row.append(hessian)
            row = torch.stack(row)
            f_ux.append(row)
        f_ux = torch.stack(f_ux)
        return f_ux
    
    def f_uu(self, x, u):
        f_u = self.f_u(x, u)
        f_uu = []
        for i in range(f_u.shape[0]):
            row = []
            for j in range(f_u.shape[1]):
                hessian = torch.autograd.grad(f_u[i][j], u, allow_unused=True, create_graph=True)[0]
                if (hessian == None):
                    hessian = torch.zeros((1))
                row.append(hessian)
            row = torch.stack(row)
            f_uu.append(row)
        f_uu = torch.stack(f_uu)
        return f_uu
