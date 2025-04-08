import torch
import torch.nn as nn
import torch.optim as optim
import string

data = "i love deep learning#"
letters = string.ascii_lowercase + " #"
n = len(letters)

def ltt(ch):
    vec = torch.zeros(n)
    vec[letters.find(ch)] = 1
    return vec

def getLine(str):
    ans = []
    for ch in str:
        ans.append(ltt(ch))
    return torch.cat(ans,dim=0).view(1,len(str),n)

target = []
for ch in data[1:]: #pehla letter kisika target nahi hai
    target.append(letters.find(ch))
target = torch.tensor(target)

class RNN(nn.Module):
    def __init__(self,inp_size,hid_size):
        super().__init__()
        self.rnn = nn.RNN(inp_size,hid_size,batch_first=True)

    def forward(self,x,h):
        op,_ = self.rnn(x,h)
        return op

model = RNN(n,n)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.01)

ip = getLine(data[:-1]) # last letter is # returned is shape(N,L,is)
seq_len = ip.size(1) # L
epochs = 120

for ep in range(epochs):
    optimizer.zero_grad()
    hid = torch.rand(1,1,n)
    op = model(ip,hid).view(-1,n)
    loss = criterion(op,target)
    loss.backward()
    optimizer.step()
    if ep==epochs-1:
        print(loss)

def predict(str):
    ip = getLine(str)
    hid = torch.rand(1, 1, n)
    op = model(ip, hid)
    last_op = op[0][-1]
    idx = last_op.topk(1)[1].item()
    pred = letters[idx]
    return pred

print(predict("i love deep learning"))

