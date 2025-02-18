import torch
import torch.nn as nn

img = torch.rand(1,1,6,6)

conv = nn.Conv2d(1,3,3,1,0)
output = conv(img)

print("outimage=", output.shape)
print("in image=", img.shape)


import torch.nn.functional as F

img = torch.rand(1,1,6,6)
kernel = torch.rand(3,1,3,3) # [out_channels, in_channels, kernel_height, kernel_width]
op = F.conv2d(img,kernel,stride=1,padding=0)

print("op=", op.shape)
print("in image=", img.shape)
print("kernel=", kernel.shape)

# output dim = (hin - k + 2p / s) + 1
# no.para = (k*k)(in_channel*out_channel) + out_channel
