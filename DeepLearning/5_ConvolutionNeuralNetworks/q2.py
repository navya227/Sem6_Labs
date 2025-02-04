import torch
import torch.nn as nn
import torch.nn.functional as F

image = torch.rand(6, 6)
print("Original Image:")
print(image)
print("Image shape:", image.shape)

image = image.unsqueeze(dim=0).unsqueeze(dim=0)
print("\nImage after adding batch and channel dimensions:")
print(image)
print("Image shape:", image.shape)

conv2d_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=0, bias=False)

output_conv2d = conv2d_layer(image)
print("\nOutput of torch.nn.Conv2d:")
print(output_conv2d)
print("Output shape of torch.nn.Conv2d:", output_conv2d.shape)

weights = conv2d_layer.weight
print("\nKernel weights of Conv2d layer:")
print(weights)

output_conv2d_func = F.conv2d(image, weights, stride=1, padding=0)
print("\nOutput of torch.nn.functional.conv2d:")
print(output_conv2d_func)
print("Output shape of torch.nn.functional.conv2d:", output_conv2d_func.shape)

print("\nAre both outputs equal?")
print(torch.allclose(output_conv2d, output_conv2d_func))
