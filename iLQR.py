import numpy as np
import torch
from dynamics import Dynamics
from cost import Cost
import math

class iLQR():
    def __init__(self, dynamics, cost_fn, state_0, goal_state, num_steps=100, max_iter=100):
        self.num_steps = num_steps
        self.max_iter = max_iter
        self.goal_state = goal_state
        self.u_dim = 1
        self.x_dim = 6
        self.state_0 = state_0
        self.dynamics = dynamics
        self.cost_fn = cost_fn
        self.mu = 1.0
        self.mu_min = 1e-6
        self.mu_max = 1e10
        self.delta_0 = 1.00
        self.delta = self.delta_0
        # state is defined by [x, theta1, theta2, x_dot, theta1_dot, theta2_dot]
    

    def ilqr(self, x_0, U):
        # U = 20*torch.rand((self.num_steps, self.u_dim))-10   #initialize random actions
        #rollout the dynamics
        X = self.dynamics.rollout(x_0, U)
        for i in range(X.shape[0]):
            X[i,:][1] = wrapToPI(X[i,:][1])
            X[i,:][2] = wrapToPI(X[i,:][2])
        J = self.cost_fn.total_cost(X, self.goal_state, U)
        print("first cost:", J.item())

        for i in range(self.max_iter):
            # print("iteration: ", i)

            # #reset regularization
            # self.mu = 1.0
            # self.delta = self.delta_0

            # backwards pass
            x_f = X[-1,:].clone().detach().requires_grad_(True)

            V_x = self.cost_fn.l_x(x_f, self.goal_state)
            V_xx = self.cost_fn.l_xx(x_f, self.goal_state)

            # start at end and work backwards
            k_list = []
            K_list = []
            t = self.num_steps-1
            while(t>=0):
                x_t = X[t,:].clone().detach().requires_grad_(True)
                u_t = U[t,:].clone().detach().requires_grad_(True)

                I = torch.eye(self.x_dim)
                f_x = self.dynamics.f_x(x_t, u_t)
                f_u = self.dynamics.f_u(x_t, u_t)
                f_xx = self.dynamics.f_xx(x_t, u_t)
                f_ux = self.dynamics.f_ux(x_t, u_t)
                f_uu = self.dynamics.f_uu(x_t, u_t)
                l_x = self.cost_fn.l_x(x_t, self.goal_state, u_t)
                l_u = self.cost_fn.l_u(x_t, self.goal_state, u_t)
                l_uu = self.cost_fn.l_uu(x_t, self.goal_state, u_t).squeeze().squeeze()
                l_ux = self.cost_fn.l_ux(x_t, self.goal_state, u_t)
                l_xx = self.cost_fn.l_xx(x_t, self.goal_state, u_t)
                Q_uu =  l_uu \
                        + torch.t(f_u) @ (V_xx.squeeze(0) + self.mu*I) @ f_u \
                        + torch.dot(V_x.squeeze(0), f_uu.squeeze(-1).squeeze(-1))
                Q_ux = l_ux \
                        + torch.t(f_u) @ (V_xx.squeeze(0) + self.mu*I) @ f_x \
                        + torch.tensordot(V_x, f_ux)
                Q_u = l_u \
                        + torch.t(f_u) @ torch.t(V_x)
                Q_x = torch.t(l_x) + torch.t(f_x) @ torch.t(V_x)
                Q_xx = l_xx \
                        + torch.t(f_x) @ V_xx.squeeze(0) @ f_x \
                        + torch.tensordot(V_x, f_xx, dims=1)

                # regularization
                if (Q_uu > 0):
                    self.delta = min(1/self.delta_0, self.delta/self.delta_0)
                    if (self.mu * self.delta > self.mu_min):
                        self.mu = self.mu * self.delta
                    else:
                        self.mu = 0
                    if (self.mu> self.mu_max):
                        print("exceeded max regularization term")
                        break
                else:     #non-PD Q_uu
                    print("non PD Q")
                    self.delta = max(self.delta_0, self.delta * self.delta_0)
                    self.mu = max(self.mu_min, self.mu * self.delta)
                    continue
                t -= 1

                k = -1 * torch.inverse(Q_uu) @ Q_u
                K = -1 * torch.inverse(Q_uu) @ Q_ux
                K = K.squeeze(0)
                k_list.insert(0, k)
                K_list.insert(0, K)

                # update V_x and V_xx
                V_x = Q_x \
                        + torch.t(K) @ Q_uu @ k \
                        + torch.t(K) @ Q_u \
                        + torch.t(Q_ux.squeeze(0)) @ k
                V_x = torch.t(V_x)
                
                V_xx = Q_xx \
                         + torch.t(K) @ Q_uu @ K \
                         + torch.t(K) @ Q_ux \
                         + torch.t(Q_ux.squeeze(0)) @ K
            
            # forward pass
            X_hat = torch.zeros_like(X)
            U_hat = torch.zeros_like(U)
            X_hat[0,:] = X[0,:]
            J_new = None
            alphas = 1.1**(-np.arange(11)**2)
            converged = False
            for alpha in alphas:
                X_hat = torch.zeros_like(X)
                U_hat = torch.zeros_like(U)
                X_hat[0,:] = X[0,:]
                for j in range(self.num_steps):
                    U_hat[j,:] = U[j,:] + alpha*k_list[j] + K_list[j] @ (X_hat[j,:] - X[j,:])
                    X_hat[j+1,:] = self.dynamics.f(X_hat[j,:], U_hat[j,:])
                    X_hat[j+1,:][1] = wrapToPI(X_hat[j+1,:][1])
                    X_hat[j+1,:][2] = wrapToPI(X_hat[j+1,:][2])
                J_new = self.cost_fn.total_cost(X_hat, self.goal_state, U_hat)
                if (J_new < J):
                    #accept this alpha
                    break
            if (J_new > J or math.isnan(J_new) or (torch.abs(J_new-J) < 0.0001)):
                # print("Cost: ", J.item())
                # print(X[-1,:])
                self.mu = 1.0
                self.mu_min = 1e-6
                self.mu_max = 1e10
                self.delta_0 = 1.00
                self.delta = self.delta_0
                print("Final Cost: ", J.item())
                return U

            # update states and actions
            J = J_new
            U = U_hat
            X = X_hat
        self.mu = 1.0
        self.mu_min = 1e-6
        self.mu_max = 1e10
        self.delta_0 = 1.00
        self.delta = self.delta_0
        print("Final Cost: ", J.item())
        return U

def wrapToPI(phase):
    x_wrap = phase% (2 * torch.pi)
    while abs(x_wrap) > torch.pi:
        x_wrap -= 2 * torch.pi * (x_wrap/abs(x_wrap))
    return x_wrap

def main():
    state_0 = torch.ones((6), requires_grad=True)
    goal_state = torch.zeros((6))
    double_pend_dynamics = Dynamics(6, 1)
    cost_fn = Cost()
    controller = iLQR(double_pend_dynamics, cost_fn, state_0, goal_state)
    controller.ilqr()

if __name__ == "__main__":
    main()