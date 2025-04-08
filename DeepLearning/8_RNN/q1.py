import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("daily.csv")
df = df.dropna()
y = df['Price'].values
y = (y - y.min()) / (y.max() - y.min())
minm, maxm = y.min(), y.max()

seq_len = 10

X, Y = [], []

for i in range(0, 5900):
    X.append(y[i:i + seq_len])
    Y.append(y[i + seq_len])

X = np.array(X)
Y = np.array(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=False)


class TimeSeries(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


train_loader = DataLoader(TimeSeries(x_train, y_train), batch_size=256, shuffle=True)


class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(1, 5, batch_first=True)
        self.fc = nn.Linear(5, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(torch.relu(out[:, -1]))


model = RNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for ep in range(1500):
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb.view(-1, seq_len, 1)).view(-1)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
    if ep % 50 == 0:
        print(f"Epoch {ep} | Loss: {loss.item():.6f}")

test_set = TimeSeries(x_test, y_test)
pred = model(test_set[:][0].view(-1, seq_len, 1)).view(-1)

plt.plot(pred.detach().numpy(), label='predicted')
plt.plot(test_set[:][1].view(-1), label='actual')
plt.legend()
plt.show()

# Undo normalization
y_true = y * (maxm - minm) + minm
y_pred = pred.detach().numpy() * (maxm - minm) + minm
plt.plot(y_true)
plt.plot(range(len(y_true) - len(y_pred), len(y_true)), y_pred)
plt.show()
