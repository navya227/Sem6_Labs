import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import string

# Prepare data
data = "i love deep learning#"
letters = string.ascii_lowercase + " #"
n = len(letters)  # total classes (input_size == output_size)

def ltt(ch):
    vec = torch.zeros(n)
    vec[letters.find(ch)] = 1
    return vec

def getLine(str_seq):
    return torch.stack([ltt(ch) for ch in str_seq], dim=0)  # shape: [seq_len, vocab]

class CharDataset(Dataset):
    def __init__(self, text, input_len=2, target_len=3):
        self.input_len = input_len
        self.target_len = target_len
        self.text = text
        self.samples = []

        for i in range(len(text) - input_len - target_len + 1):
            input_seq = text[i : i + input_len]
            target_seq = text[i + input_len : i + input_len + target_len]
            self.samples.append((input_seq, target_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_str, target_str = self.samples[idx]
        input_tensor = getLine(input_str)  # [input_len, vocab]
        target_tensor = torch.tensor([letters.find(c) for c in target_str])  # [target_len]
        return input_tensor, target_tensor

# Hyperparameters
input_len = 2
target_len = 3
batch_size = 1
hidden_size = n
epochs = 120

# Dataset and DataLoader
dataset = CharDataset(data, input_len, target_len)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# RNN Model with batch_first=True
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        out, _ = self.rnn(x, h)         # out: [B, seq_len, hidden]
        out = self.fc(out)              # out: [B, seq_len, vocab]
        return out

model = CharRNN(n, hidden_size, n)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
for ep in range(epochs):
    for xb, yb in loader:  # xb: [B, seq_len, vocab], yb: [B, target_len]
        h = torch.zeros(1, batch_size, hidden_size)
        out = model(xb, h)                           # [B, seq_len, vocab]
        out = out[:, -target_len:, :]                # keep only last target_len outputs
        loss = criterion(out.view(-1, n), yb.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if ep % 20 == 0 or ep == epochs - 1:
        print(f"Epoch {ep}, Loss: {loss.item():.4f}")

# Prediction
def predict(input_str):
    model.eval()
    with torch.no_grad():
        x = getLine(input_str).unsqueeze(0)              # [1, input_len, vocab]
        h = torch.zeros(1, 1, hidden_size)
        out = model(x, h)                                # [1, input_len, vocab]
        last_out = out[:, -1, :]                         # [1, vocab]
        topk = torch.topk(last_out, 3).indices[0].tolist()
        return ''.join(letters[i] for i in topk)

# Test
test_input = "i "
print(f"\nInput: '{test_input}'")
print("Predicted next characters:", predict(test_input))

tot = sum(p.numel() for p in model.parameters())
print(tot)
