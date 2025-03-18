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


# Read data
def load_data():
    category_lines = {}
    all_categories = []
    file_path = "eng-fra.txt"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file {file_path} not found!")

    with open(file_path, encoding='utf-8') as f:
        lines = f.read().strip().split("\n")

    for line in lines:
        parts = line.split("\t")  # Handling tab separation
        if len(parts) == 2:
            category, name = parts
            category = unicode_to_ascii(category)
            name = unicode_to_ascii(name)
            if category not in category_lines:
                category_lines[category] = []
                all_categories.append(category)
            category_lines[category].append(name)

    return category_lines, all_categories


category_lines, all_categories = load_data()
n_categories = len(all_categories)


# Convert names to tensors
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][all_letters.index(letter)] = 1
    return tensor


# Define RNN Model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
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


n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

# Training
criterion = nn.NLLLoss()
optimizer = optim.SGD(rnn.parameters(), lr=0.005)


def random_training_example():
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


def train(category_tensor, line_tensor):
    hidden = rnn.init_hidden()
    optimizer.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()
    return output, loss.item()


# Training loop
n_iters = 1000
for iter in range(n_iters):
    category, line, category_tensor, line_tensor = random_training_example()
    output, loss = train(category_tensor, line_tensor)

    if iter % 100 == 0:
        print(f'Iteration {iter}, Loss: {loss:.4f}')


# Prediction
def predict(input_line):
    with torch.no_grad():
        hidden = rnn.init_hidden()
        line_tensor = line_to_tensor(input_line)

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        top_v, top_i = output.topk(1)
        category_index = top_i[0].item()
        return all_categories[category_index]


# Example usage
print(predict("Satoshi"))