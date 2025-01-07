import torch
import numpy as np

device = torch.device("cpu")

# q1
print(" ")
print("1")
tensor_test = torch.tensor([1, 2, 3, 4])
reshaped = torch.reshape(tensor_test,(2,2))
print("Reshaped Tensor:\n", reshaped)

t1 = torch.tensor([1, 2])
t2 = torch.tensor([3, 4])
stacked = torch.stack([t1,t2], dim=0)
print("Stacked Tensor:\n", stacked)

tensor_to_squeeze = torch.tensor([[[1], [2]]])
squeezed = tensor_to_squeeze.squeeze()
print("Squeezed Tensor:\n", squeezed)

unsqueezed = tensor_test.unsqueeze(1)
print("Unsqueezed Tensor:\n", unsqueezed)

#q2
print(" ")
print("2")
tensor = torch.randn(1, 2, 3)
print("Original Tensor:")
print(tensor)
print("Original Shape:", tensor.shape)
permuted_tensor = tensor.permute(1, 0, 2) # exention of transpose

print("\nPermuted Tensor:")
print(permuted_tensor)
print("Permuted Shape:", permuted_tensor.shape)

#q3
print(" ")
print("3")
tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(tensor[1][2])

#q4
print(" ")
print("4")
np_arr = np.arange(1,10)
tensor = torch.from_numpy(np_arr)
print(tensor)

np_arr = tensor.numpy()
print(np_arr)

#q5
print(" ")
print("5")
mul1 = torch.randn(7,7)
print(mul1)

#q6
print(" ")
print("6")
mul2 = torch.randn(1,7)
prod = torch.matmul(mul1,mul2.permute(1,0))
print(prod)

#q7,8
print(" ")
print("7,8")
mul1 = torch.randn(7,7).to(device)
mul2 = torch.randn(1,7).to(device)
prod = torch.matmul(mul1,mul2.permute(1,0))
print(prod)

#q9
print(" ")
print("9")
print(torch.max(prod))
print(torch.min(prod))

#q10
print(" ")
print("10")
print(torch.argmin(prod))
print(torch.argmax(prod))

#q11
print(" ")
print("11")
torch.seed = 7
ten = torch.randn(1,1,1,10)
print(ten)
print(ten.squeeze())



