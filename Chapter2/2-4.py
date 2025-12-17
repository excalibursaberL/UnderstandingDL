import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.1, 3, 0.05)
plt.plot(x, x ** 3 - 1 / x, 'x', label = 'f(x)')
plt.plot(x, x * 4 - 4, label = 'Tangent line(x=1)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()