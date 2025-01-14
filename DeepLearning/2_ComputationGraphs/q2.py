import torch
w = torch.tensor(1.0,requires_grad=True)
x = torch.tensor(2.0,requires_grad=True)
b = torch.tensor(1.0,requires_grad=True)

u = w*x
v = u+b
a = torch.relu(v)

a.backward()

print(w.grad)