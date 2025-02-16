import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

x = torch.tensor( [1., 5., 10., 10., 25., 50., 70., 75., 100.]).view(-1, 1)
y = torch.tensor( [0., 0., 0., 0., 0., 1., 1., 1., 1]).view(-1, 1)

lr = 0.001
epochs = 100

class RegressionDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

dataset = RegressionDataset(x, y)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

model = LogisticRegression()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

loss_list = []

for epoch in range(epochs):
    l = 0.0
    for i, (ip, tar) in enumerate(dataloader):
        yp = model(ip)
        loss = criterion(yp, tar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        l += loss.item()

    l /= len(dataloader)
    loss_list.append(l)


plt.plot(range(epochs), loss_list)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.show()

print(f"Final W (Weight): {model.linear.weight.item()}")
print(f"Final B (Bias): {model.linear.bias.item()}")
