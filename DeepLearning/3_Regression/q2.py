import torch

w = torch.tensor(1.0,requires_grad=True)
b = torch.tensor(1.0,requires_grad = True)
lr = torch.tensor(0.001)

x = torch.tensor([2.,4.])
y = torch.tensor([20.,40.])

loss_list = []

for epoch in range(2):
    loss = 0.0
    for i in range(len(x)):
        yp = x[i]*w + b
        loss += (yp-y[i])**2

    loss /= len(x)
    loss_list.append(loss)
    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()
    b.grad.zero_()

print(f"w = {w.item()}")
print(b.item())

w = torch.tensor([1.])
b = torch.tensor([1.])

for epochs in range(2):
    l = 0.0
    dw = 0.0
    db = 0.0
    for i in range(len(x)):
        yp = w * x[i] + b
        dw += 2 * (yp - y[i]) * x[i]
        db += 2 * (yp - y[i])
        l += (yp - y[i]) ** 2
    l /= len(x)
    dw /= len(x)
    db /= len(x)

    w -= lr * dw
    b -= lr * db

    print(w)
    print(b)

print(f"ANALYTICAL SOLN W : {w.item()} B : {b.item()}")
