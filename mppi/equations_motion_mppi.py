import torch


class motionModel():
    def __init__(self, mass_cart, mass_arm1, mass_arm2, length_arm1, length_arm2):
        self.m0 = mass_cart
        self.m1 = mass_arm1
        self.m2 = mass_arm2
        self.l1 = length_arm1
        self.l2 = length_arm2

    def calculateD(self, state):

        D = torch.tensor([[self.m0+ self.m1+self.m2, (0.5*self.m1 + self.m2)*self.l1*torch.cos(state[1]), 0.5*self.m2*self.l2*torch.cos(state[2])],
                  [(0.5*self.m1+self.m2)*self.l1*torch.cos(state[1]), (1/3*self.m1+self.m2)*self.l1*self.l1, 0.5*self.m2*self.l1*self.l2*torch.cos(state[1]-state[2])],
                  [0.5*self.m2*self.l2*torch.cos(state[2]), 0.5*self.m2*self.l1*self.l2*torch.cos(state[1]-state[2]), 1/3*self.m2*self.l2*self.l2]], requires_grad=True)
        return D
    
    def calculateC(self,state):
        C = torch.tensor([[0, -(0.5*self.m1+self.m2)*self.l1*torch.sin(state[1])*state[4], -0.5*self.m2*self.l2*torch.sin(state[2])*state[5]],
                      [0, 0, 0.5*self.m2*self.l1*self.l2*torch.sin(state[1] - state[2])*state[5]],
                      [0, -0.5*self.m2*self.l1*self.l2*torch.sin(state[1]-state[2])*state[4], 0]], requires_grad=True)
        return C
    
    def calculateG(self,state):
        g = 9.8
        G = torch.tensor([[0],
                          [-0.5*(self.m1+self.m2)*self.l1*g*torch.sin(state[1])],
                          [-0.5*self.m2*g*self.l2*torch.sin(state[2])]], requires_grad=True)
        return G

    def f(self, state, action):
        dt = 0.05
        D = self.calculateD(state)
        D_inv = D.inverse()
        C = self.calculateC(state)
        G = self.calculateG(state)
        H = torch.tensor([[1.0],[0.0],[0.0]])

        A_zeros = torch.zeros([6,3])
        A_values = torch.cat([torch.eye(3), torch.matmul(-D_inv, C)], dim=0)
        A = torch.cat([A_zeros, A_values], dim=1)
        #print(torch.matmul(D_inv, H))
        #print(torch.zeros([3,3]))
        B = torch.cat([torch.zeros([3]), torch.matmul(D_inv, H).squeeze()], dim=0)
        L = torch.cat([torch.zeros([3]), torch.matmul(-D_inv, G).squeeze()], dim=0)

        #print(torch.matmul(A, state).shape)
        #print((B * action).shape)
        #print(L.shape)
        state_dot = torch.matmul(A, state) + B * action + L
        #print(state_dot.shape)
        next_state = state + state_dot * dt
        #print(next_state.shape)

        return next_state
    
