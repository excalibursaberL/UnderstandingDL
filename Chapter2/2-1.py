import torch

x = torch.randn((2, 3, 4))
y = torch.randn((2, 3, 4))
z = x * y
print(z[-1])