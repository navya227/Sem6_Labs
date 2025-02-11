import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets
import os


# question3
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Saves the checkpoint to a file."""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def train_model_with_checkpoint(model, train_loader, val_loader, criterion, optimizer, num_epochs=5,
                                checkpoint_interval=1):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy * 100:.2f}%")

        if accuracy > best_acc:
            best_acc = accuracy
            print(f"New best accuracy: {accuracy * 100:.2f}%")
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss / len(train_loader),
                'best_acc': best_acc
            }, filename="best_model.pth.tar")

        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss / len(train_loader),
                'best_acc': best_acc
            }, filename=f"checkpoint_epoch_{epoch + 1}.pth.tar")


data_dir = 'cats_and_dogs_filtered'

transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform['train'])
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'validation'), transform['val'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = models.alexnet(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

train_model_with_checkpoint(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, checkpoint_interval=1)

