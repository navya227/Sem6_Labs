import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(1,64,3),
            nn.ReLU(),
            nn.MaxPool2d((2,2),stride=2),
            nn.Conv2d(64,128,3),
            nn.ReLU(),
            nn.MaxPool2d((2,2),stride=2),
            nn.Conv2d(128,64,3),
            nn.ReLU(),
            nn.MaxPool2d((2,2),stride=2),
        )
        self.linear = nn.Sequential(
            nn.Linear(64,20,bias=True),
            nn.ReLU(),
            nn.Linear(20,10,bias=True)
        )

    def forward(self,x):
        x=self.net(x)
        x = x.flatten(start_dim=1)
        x=self.linear(x)
        return x

batch_size = 32
mnist_trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)

mnist_testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(mnist_testset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.save(model,"../06_TransferLearning/model.pt")
# torch.save(model.state_dict(), '../06_TransferLearning/mnist_stateDict.pt')

model = MNIST_CNN()
model.load_state_dict(torch.load('./mnist_stateDict.pt'))
model.to(device)

optimizer = optim.Adam(model.linear.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# fine-tuning
# Freeze CNN layers
for param in model.net.parameters():
    param.requires_grad = False

# Ensure linear layers are trainable
for param in model.linear.parameters():
    param.requires_grad = True

epochs = 1

model.train()
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

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
# torch.save({
#         'epoch': epoch + 1,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss.item()
#     }, '../06_TransferLearning/checkpoint.pth')

checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']

print(f"Resuming from epoch {start_epoch}...")

model.eval()
correct = 0
total = 0
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

accuracy = 100.0 * correct / total
print(f"The overall accuracy is {accuracy:.2f}%")

print("\nModel's state_dict:")
for param_tensor in model.state_dict().keys():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
