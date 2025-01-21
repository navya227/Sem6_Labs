import torch
import matplotlib.pyplot as plt

X = torch.tensor([12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4,
                  19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4, 19.5, 19.7, 21.2])

Y = torch.tensor([11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6,
                  16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2, 17.0, 17.2, 18.6])

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

def forward(x):
    return w * x + b

# Mean Square Error.
def criterion(Y_pred, y):
    return torch.mean((Y_pred - y) ** 2)

step_size = 0.001
iter = 200
loss_list = []

for i in range(iter):
    Y_pred = forward(X)
    loss = criterion(Y_pred, Y)
    loss_list.append(loss.item())

    loss.backward()

    w.data -= step_size * w.grad.data
    b.data -= b.data - step_size * b.grad.data

    # necesary otherwise the gradients from previous iterations will be added to the gradients computed in the current iteration
    w.grad.data.zero_()
    b.grad.data.zero_()

Y_pred_final = forward(X)

plt.scatter(X.numpy(), Y.numpy(), label='Data Points')
plt.plot(X.numpy(), Y_pred_final.detach().numpy(), color='orange', label='Fitted Line')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()


plt.plot(loss_list, 'r')
plt.tight_layout()
plt.xlabel("Epochs/Iterations")
plt.ylabel("Loss")
plt.show()


