import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal  # for sawtooth waveform

# Generate sawtooth wave data
x_vals = np.linspace(0, 10, 200)  # covers several cycles
y_vals = signal.sawtooth(2 * np.pi * x_vals)  # sawtooth function

# Prepare sequences
seq_len = 10
X = []
Y = []
for i in range(len(y_vals) - seq_len):
    X.append(y_vals[i:i + seq_len])
    Y.append([y_vals[i + seq_len]])  # keep shape [1]

X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
Y = torch.tensor(Y, dtype=torch.float32)

# Custom dataset
class myData(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

dataset = myData(X, Y)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, out_size=2):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]) # final output shape: [batch, 2]

model = RNNModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

losses = []
epochs = 100
for epoch in range(epochs):
    total_loss = 0
    for xb, yb in dataloader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    losses.append(avg_loss)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")


plt.plot(losses)
plt.title("Epoch vs Loss (")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()


model.eval()
with torch.no_grad():
    predictions = model(X)

plt.plot(Y.numpy(), label='Actual')
plt.plot(predictions.numpy(), label='Predicted')
plt.title("Sawtooth Prediction")
plt.legend()
plt.grid(True)
plt.show()
