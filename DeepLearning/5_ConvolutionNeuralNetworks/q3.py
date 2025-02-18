import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1,64,3),
                                 nn.ReLU(),
                                 nn.MaxPool2d((2,2),stride = 2),
                                 nn.Conv2d(64, 128, 3),
                                 nn.ReLU(),
                                 nn.MaxPool2d((2, 2), stride=2),
                                 nn.Conv2d(128, 64, 3),
                                 nn.ReLU(),
                                 nn.MaxPool2d((2, 2), stride=2)
                                 )
        self.linear = nn.Sequential(nn.Linear(64,20,bias=True),
                                    nn.ReLU(),
                                    nn.Linear(20,10,bias=True)
                                    )
    def forward(self,x):
        x = self.net(x)
        x = x.flatten(start_dim = 1)
        x = self.linear(x)
        return x

trainset = datasets.MNIST(root='./data/',train=True,download=True,transform=ToTensor())
trainloader = DataLoader(trainset,batch_size=32,shuffle=True)

testset = datasets.MNIST(root='./data/',train=False,download=True,transform=ToTensor())
testloader = DataLoader(testset,batch_size=32,shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MNIST_CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr = 0.01)

loss_list = []
epochs = 1

for epoch in range(epochs):
    l = 0.0
    for i,(inp,tar) in enumerate(trainloader):
        inp = inp.to(device)
        tar = tar.to(device)

        optimizer.zero_grad()
        yp = model(inp)
        loss = criterion(yp,tar)
        loss.backward()
        optimizer.step()
        l+= loss.item()

    l/=len(trainloader)
    loss_list.append(l)
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item()
    }, '../06_TransferLearning/checkpoint.pth')

plt.plot(list(range(epochs)),loss_list)
plt.show()

model.eval()
all_pred = []
all_label = []
correct = 0
total = 0
with torch.no_grad():
    for i, (inp, tar) in enumerate(testloader):
        inp = inp.to(device)
        tar = tar.to(device)

        op = model(inp)
        _,pred = torch.max(op,1)
        all_pred.extend(pred.cpu().numpy())
        all_label.extend(tar.cpu().numpy())
        correct += (pred==tar).sum().item()
        total += tar.size(0)

accuracy = correct/total
print(f"Accuracy = {accuracy}")

conf_matrix = confusion_matrix(all_label, all_pred)
print('Confusion Matrix:')
print(conf_matrix)

num_para = sum(p.numel() for p in model.parameters())
print(f"Total Parameters = {num_para}")

torch.save(model,"../06_TransferLearning/model.pt")
torch.save(model.state_dict(), '../06_TransferLearning/mnist_stateDict.pt')
