import matplotlib.pyplot as plt
import torch

x = torch.arange(0, 3, 0.01)
x.requires_grad_(True)
x1 = x.detach()
y1 = torch.sin(x1)
y = torch.sin(x)
y.sum().backward()
plt.plot(x1, y1, label = 'sin(x)')
plt.plot(x1, x.grad, label = 'cox(x)')
plt.legend()
plt.show()