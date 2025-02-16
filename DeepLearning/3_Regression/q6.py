import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

x1 = torch.tensor([3.,4.,5.,6.,2.]).view(-1,1)
x2 = torch.tensor([8.,5.,7.,3.,1.]).view(-1,1)
x = torch.cat((x1,x2),dim=1)
y = torch.tensor([-3.7,3.5,2.5,11.5,5.7]).view(-1,1)
lr = 0.001
epochs = 100

class myData(Dataset):
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx],self.Y[idx]

dataset = myData(x,y)
dataloader = DataLoader(dataset,batch_size=1,shuffle=True)

class LinMod(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2,1)
    def forward(self,x):
        return self.linear(x)

model = LinMod()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr = lr)

loss_list = []

for epoch in range(epochs):
    l= 0.0
    for i,(ip,tar) in enumerate(dataloader):
        yp = model(ip)
        loss = criterion(yp,tar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        l+=loss.item()

    l/=len(x)
    loss_list.append(l)

print(f"Final W (Weight): {model.linear.weight.data}")
print(f"Final B (Bias): {model.linear.bias.item()}")

print("Predicted output for 3,2")
test=torch.tensor([(3.,2.)])
print(model(test).item())

plt.plot(range(epochs), loss_list)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.show()

