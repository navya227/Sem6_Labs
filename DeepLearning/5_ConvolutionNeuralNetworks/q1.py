import torch
import torch.nn.functional as F

image = torch.rand(6, 6)
print("Original Image:")
print(image)
print("Image shape:", image.shape)

image = image.unsqueeze(dim=0)
print("\nImage after unsqueeze to add batch dimension:")
print(image)
print("Image shape:", image.shape)

image = image.unsqueeze(dim=0)
print("\nImage after unsqueeze to add channel dimension:")
print(image)
print("Image shape:", image.shape)

kernel = torch.ones(3, 3)
print("\nKernel before reshaping:")
print(kernel)
print("Kernel shape:", kernel.shape)

kernel = kernel.unsqueeze(dim=0)
kernel = kernel.unsqueeze(dim=0)
print("\nKernel after reshaping:")
print(kernel)
print("Kernel shape:", kernel.shape)

outimage_stride_1 = F.conv2d(image, kernel, stride=1, padding=0)
print("\nOutput image with stride=1 and padding=0:")
print(outimage_stride_1)
print("Output image shape with stride=1 and padding=0:", outimage_stride_1.shape)

outimage_stride_2 = F.conv2d(image, kernel, stride=2, padding=0)
print("\nOutput image with stride=2 and padding=0:")
print(outimage_stride_2)
print("Output image shape with stride=2 and padding=0:", outimage_stride_2.shape)

outimage_padding_1 = F.conv2d(image, kernel, stride=1, padding=1)
print("\nOutput image with stride=1 and padding=1:")
print(outimage_padding_1)
print("Output image shape with stride=1 and padding=1:", outimage_padding_1.shape)

H, W = 6, 6
K_H, K_W = 3, 3
S = 1
P = 0

out_height = (H + 2 * P - K_H) // S + 1
out_width = (W + 2 * P - K_W) // S + 1
print("\nManually calculated output size with stride=1 and padding=0:")
print(f"Output Height: {out_height}, Output Width: {out_width}")

num_parameters = (1 * K_H * K_W + 1) * 1
print("\nTotal number of parameters in the network:")
print(f"Number of parameters: {num_parameters}")
