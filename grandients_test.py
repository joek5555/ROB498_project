from dynamics import Dynamics
import torch

dynamics = Dynamics(1, 1)

x = torch.tensor([[2.0]], requires_grad=True)
u = torch.tensor([[3.0]], requires_grad=True)
f = dynamics.f(x,u)
f_x = dynamics.f_x(x,u)
f_u = dynamics.f_u(x,u)
f_xu = dynamics.f_xu(x,u)
f_xx = dynamics.f_xx(x,u)
f_ux = dynamics.f_ux(x,u)
f_uu = dynamics.f_uu(x,u)

print("f: ", f)
print("f_x: ", f_x)
print("f_u: ", f_u)
print("f_xu: ", f_xu)
print("f_xx: ", f_xx)
print("f_ux: ", f_ux)
print("f_uu: ", f_uu)