import torch

X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)

l1_weight = torch.tensor([[0.5183, 0.5773], [-0.1521, 0.6407]])
l1_bias = torch.tensor([-0.1575, 0.1296])
l2_weight = torch.tensor([[-0.3766, 0.3448]])
l2_bias = torch.tensor([0.5208])

l1_output = X @ l1_weight.T + l1_bias
l1_activated = torch.relu(l1_output)

l2_output = l1_activated @ l2_weight.T + l2_bias

for i in range(len(X)):
    print(f"Input {X[i].tolist()} => Output {l2_output[i].item()}")
