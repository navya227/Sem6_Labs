import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

loss_list = []
torch.manual_seed(42)

x = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([0,1,1,0], dtype=torch.float32).view(-1,1)
lr = 0.001

device = "cuda" if torch.cuda.is_available() else "cpu"
class myData (Dataset):
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx],self.Y[idx]

dataset = myData(x,y)
dataloader = DataLoader(dataset,batch_size=1,shuffle=True)

class XORModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2,2,bias=True)
        self.a1 = nn.Sigmoid()
        self.l2 = nn.Linear(2,1,bias=True)

    def forward(self,x):
        x = self.l1(x)
        x = self.a1(x)
        x = self.l2(x)
        return x

model = XORModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr = lr)

epochs = 1000
loss_list = []

for epoch in range(epochs):
    l = 0.0
    for i,(ip,tar) in enumerate(dataloader):
        yp = model(ip)
        loss = criterion(yp,tar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        l+=loss.item()
    l/=len(x)
    loss_list.append(l)

model.eval()
with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloader):
        outputs = model(inputs)
        print(inputs, outputs)

for param in model.named_parameters():
    print(param)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

plt.plot(list(range(epochs)), loss_list)
plt.show()

# model.eval()
# correct = 0
# total = 0

# with torch.no_grad():
#     for i, (inputs, labels) in enumerate(dataloader):
#         inputs, labels = inputs.to(device), labels.to(device)

#         outputs = model(inputs)
#         predicted = torch.round(outputs)  # Round to 0 or 1

#         correct += (predicted == labels).sum().item()
#         total += labels.size(0)

# accuracy = correct / total * 100
# print(f"Accuracy: {accuracy:.2f}%")
