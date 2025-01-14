import torch

x = torch.tensor(1.0, requires_grad=True)
f = torch.exp(-x**2 - 2*x - torch.sin(x))
f.backward()

ana = -torch.exp(-x**2 - 2*x - torch.sin(x))*(2*x+2+torch.cos(x))
print("By pytorch:", x.grad.item())
print("Analytically: ",ana.item())
