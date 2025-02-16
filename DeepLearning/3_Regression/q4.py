import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim

x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0]).view(-1,1)
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0]).view(-1,1)
lr = torch.tensor(0.001)

class myData (Dataset):
    def __init__(self,X,Y):
        self.X = torch.tensor(X,dtype=torch.float32)
        self.Y = torch.tensor(Y,dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx],self.Y[idx]

dataset = myData(x,y)
dataloader = DataLoader(dataset,batch_size=1,shuffle=True)

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.))
        self.b = nn.Parameter(torch.tensor(1.))

    def forward(self,x):
        return self.w * x + self.b

model = LinearModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

loss_list = []

epochs = 100
for epoch in range(epochs):
    epoch_loss = 0.0
    for i, (inputs, targets) in enumerate(dataloader):

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    loss_list.append(avg_loss)

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

print(f"Final W (Weight): {model.w.item()}")
print(f"Final B (Bias): {model.b.item()}")
