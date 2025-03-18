import torch
import torch.nn as nn
import torch.optim as optim
import os
import unicodedata
import string
import random

# Character set
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


# Helper function to convert Unicode to ASCII
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if c in all_letters
    )


# Define RNN Model for Next Character Prediction
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


# Initialize model
n_hidden = 128
char_rnn = CharRNN(n_letters, n_hidden, n_letters)


# Convert character to tensor
def char_to_tensor(char):
    tensor = torch.zeros(1, n_letters)
    tensor[0][all_letters.index(char)] = 1
    return tensor  # Remove extra unsqueeze


# Convert line to tensor
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][all_letters.index(letter)] = 1
    return tensor


# Training setup
criterion = nn.NLLLoss()
optimizer = optim.SGD(char_rnn.parameters(), lr=0.005)


def train(input_line_tensor, target_line_tensor):
    hidden = char_rnn.init_hidden()
    optimizer.zero_grad()

    loss = 0
    for i in range(input_line_tensor.size(0)):
        output, hidden = char_rnn(input_line_tensor[i], hidden)
        loss += criterion(output, target_line_tensor[i].unsqueeze(0))

    loss.backward()
    optimizer.step()
    return loss.item() / input_line_tensor.size(0)


# Training loop
n_iters = 50000
data = ["hello", "world", "python", "torch", "character", 'type']
for iter in range(n_iters):
    word = random.choice(data)
    input_tensor = line_to_tensor(word[:-1])
    target_tensor = torch.tensor([all_letters.index(c) for c in word[1:]], dtype=torch.long)
    loss = train(input_tensor, target_tensor)

    if iter % 5000 == 0:
        print(f'Iteration {iter}, Loss: {loss:.4f}')


# Predict next character
def predict_next(input_char, hidden):
    input_tensor = char_to_tensor(input_char)  # No need to unsqueeze
    output, hidden = char_rnn(input_tensor, hidden)
    top_v, top_i = output.topk(1)
    predicted_index = top_i[0].item()
    return all_letters[predicted_index], hidden


# Generate a sequence
def generate(start_char, length=10):
    hidden = char_rnn.init_hidden()
    output_str = start_char

    for _ in range(length):
        next_char, hidden = predict_next(output_str[-1], hidden)
        output_str += next_char

    return output_str


# Example usage
print(generate("to"))
