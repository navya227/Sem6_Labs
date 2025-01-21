import torch
import matplotlib.pyplot as plt

X = torch.tensor([2,4])

Y = torch.tensor([20,40])

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

step_size = 0.001
epochs = 2
loss_list = []

for epoch in range(epochs):
    Y_pred = w * X + b

    loss = torch.mean((Y_pred - Y) ** 2)
    loss_list.append(loss.item())

    loss.backward()

    print(f"Epoch {epoch + 1}:")
    print(f"  w.grad: {w.grad.item()}")
    print(f"  b.grad: {b.grad.item()}")  #

    with torch.no_grad():
        w -= step_size * w.grad
        b -= step_size * b.grad

    w.grad.zero_()
    b.grad.zero_()

    print(f"  Updated w: {w.item()}")
    print(f"  Updated b: {b.item()}")
    print('-' * 40)

plt.plot(loss_list, 'r')
plt.xlabel("Epochs/Iterations")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.show()
