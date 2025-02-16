import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

x = torch.tensor( [12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4,
19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4, 19.5, 19.7, 21.2]).view(-1,1)
y = torch.tensor( [11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6,
16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2, 17.0, 17.2, 18.6]).view(-1,1)
lr = 0.001

class myData(Dataset):
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        return self.X[idx] , self.Y[idx]

dataset = myData(x,y)
dataloader = DataLoader(dataset,batch_size=1,shuffle=True)

class LinMod(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self,x):
        return self.linear(x)

model = LinMod()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=lr)

loss_list = []

for epoch in range(100):
    l = 0.0
    for i,(ip,tar) in enumerate(dataloader):
        yp = model(ip)
        loss = criterion(yp,tar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        l += loss.item()
    l/=len(x)
    loss_list.append(l)

print(f"Final W (Weight): {model.linear.weight.item()}")
print(f"Final B (Bias): {model.linear.bias.item()}")


