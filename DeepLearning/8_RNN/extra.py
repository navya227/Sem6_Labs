import torch
import torch.nn as nn
import torch.optim as optim

seq_len = 5
batch_size = 4
input_size = 1
hidden_size = 16
output_size = 1


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        out, _ = self.rnn(x, h)      # out: (batch, seq_len, hidden_size)
        out = self.fc(out)           # map to output: (batch, seq_len, output_size)
        return out

inputs = torch.tensor([1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], dtype=torch.float32).view(batch_size, seq_len, 1)

targets = torch.tensor([
    [2, 3, 4, 5, 6],
    [7, 8, 9,10,11],
    [12,13,14,15,16],
    [17,18,19,20,21]
], dtype=torch.float32).view(batch_size, seq_len, 1)

model = RNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 300
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")


def predict_next(seq):
    model.eval()
    with torch.no_grad():
        inp = torch.tensor(seq, dtype=torch.float32).view(1, seq_len, 1)
        h = torch.rand(1,1,hidden_size)
        out = model(inp,h)
        next_val = out[0][-1].item() # last time step's output
        return next_val

# Try prediction
test_seq = [1,2,3,4,5]
predicted = predict_next(test_seq)
print(f"\nInput sequence: {test_seq}")
print(f"Predicted next number: {predicted}")
