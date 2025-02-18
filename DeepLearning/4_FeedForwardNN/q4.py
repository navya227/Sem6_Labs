import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision.datasets as datasets
from sklearn.metrics import confusion_matrix
from torchvision.transforms import ToTensor

torch.manual_seed(42)

batch_size = 32
trainset = datasets.MNIST(root='./data/', train=True, download=True, transform=ToTensor())
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = datasets.MNIST(root='./data/', train=False, download=True, transform=ToTensor())
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

class FFN(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(28 * 28, 100, bias=True),
			nn.Linear(100, 100, bias=True),
			nn.Linear(100, 10, bias=True),
			nn.ReLU()
		)
	# 	number of para = [(28x28x100)+100]+[(100x100]+100+[(100x10)+10]

	def forward(self,x):
		x=x.view(-1,28*28)
		x = self.net(x)
		return x

device = "cuda" if torch.cuda.is_available() else "cpu"
model=FFN().to(device)

criterion=torch.nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01)
loss_list=[]

epochs = 5
for epoch in range(epochs):
	epoch_loss=0
	for i,(inputs,labels) in enumerate(trainloader):
		inputs=inputs.to(device)
		labels=labels.to(device)

		optimizer.zero_grad()
		outputs=model(inputs)
		loss=criterion(outputs,labels)
		loss.backward()
		optimizer.step()
		epoch_loss+=loss

	epoch_loss/=len(trainloader)
	print(f"Epoch {epoch+1} loss : {epoch_loss.item()}")
	loss_list.append(epoch_loss.item())

plt.plot(list(range(epochs)),loss_list)
plt.show()

model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
		for inputs, labels in testloader:
			inputs, labels = inputs.to(device), labels.to(device)
			outputs = model(inputs)
			_, predicted = torch.max(outputs, 1)
			all_preds.extend(predicted.cpu().numpy())
			all_labels.extend(labels.cpu().numpy())

conf_matrix = confusion_matrix(all_labels, all_preds)
print('Confusion Matrix:')
print(conf_matrix)

num_params = sum(p.numel() for p in model.parameters())
print(f"Total number of learnable parameters in the model: {num_params}")
