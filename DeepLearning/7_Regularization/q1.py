import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import zipfile
import urllib.request
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = 'cats_and_dogs_filtered'

if not os.path.exists(data_dir):
    print("Downloading Cats and Dogs dataset...")
    dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
    dataset_path = "cats_and_dogs_filtered.zip"

    urllib.request.urlretrieve(dataset_url, dataset_path)

    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall()

    print(f"Dataset extracted to {data_dir}")

# Define the model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)  # Adjusted for 256x256 image size
        self.fc2 = nn.Linear(512, 2)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 32 * 32)  # Correct flatten size
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load dataset
train_dir = os.path.join(data_dir, 'train')
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model, criterion, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)  # L2 regularization via weight_decay

# Training with optimizer's weight_decay
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader)}")

# Training with manual L2 regularization
print("With L2 norm:")
l2_lambda = 0.01  # Regularization strength
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Calculate L2 regularization manually
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.norm(param, 2)  # L2 norm of parameters

        loss += l2_lambda * l2_reg  # Add L2 penalty to loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader)}")
