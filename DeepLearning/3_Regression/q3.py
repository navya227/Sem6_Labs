import torch

x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0])
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])
lr = torch.tensor(0.001)

class RegressionModel:
    def __init__(self):
        self.w = torch.tensor(1.0,requires_grad=True)
        self.b = torch.tensor(1.0, requires_grad=True)

    def forward(self,x):
        return x * self.w + self.b

    def update(self):
        self.w -= lr * self.w.grad
        self.b -= lr * self.b.grad

    def reset_grad(self):
        self.w.grad.zero_()
        self.b.grad.zero_()

    def criterion(self,yp,y):
        return (yp-y)**2

model = RegressionModel()
loss_list = []

for epoch in range(100):
    loss = 0.0
    for i in range(len(x)):
        yp = model.forward(x[i])
        loss += model.criterion(yp,y[i])

    loss/= len(x)
    loss_list.append(loss)
    loss.backward()

    with torch.no_grad():
        model.update()

    model.reset_grad()

print(model.w.item())
print(model.b.item())
print(loss.item())
