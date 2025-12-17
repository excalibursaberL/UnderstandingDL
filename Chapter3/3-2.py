import random
import torch
from matplotlib import pyplot as plt

def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

# print('features:', features[0], '\nlabels:', labels[0])
# plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
# plt.show()

def data_iter(batch_size, features, labels):
    """创建迭代器，逐批次读取数据集"""
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    """均方损失函数"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    """ 小批次 SGD """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 生成含有噪声的 y = Xw + b
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# 初始化模型参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 训练
lr = 0.03
num_epochs = 3
batch_size = 10
net = linreg
loss = squared_loss

# 设置绘图
fix, axes = plt.subplots(1, num_epochs)

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size=batch_size, features = features, labels = labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        # 使用参数梯度更新参数
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        y_hat = net(features, w, b)
        train_l = loss(y_hat, labels)

        ax = axes[epoch] if num_epochs > 1 else axes

        n_show = 100
        indices = torch.randperm(len(features))[:n_show]
        sorted_indices = torch.argsort(features[indices, 1])

        x_display = features[indices, 1][sorted_indices].detach().numpy()
        y_true_display = labels[indices][sorted_indices].detach().numpy().flatten()
        y_hat_display = y_hat[indices][sorted_indices].detach().numpy().flatten()

        ax.scatter(x_display, y_true_display, alpha=0.5, color='blue', label='True data')
        ax.scatter(x_display, y_hat_display, alpha=0.5, color='red', label='Predicted data')

        for i in range(len(x_display)):
            ax.plot([x_display[i], x_display[i]], [y_true_display[i], y_hat_display[i]],'k-', alpha=0.2, linewidth = 0.5)
        ax.set_title(f'epoch: {epoch}, loss: {float(train_l.mean()):f}')

plt.legend()
plt.tight_layout()
plt.show()
print(f"w 的估计误差:{true_w - w.reshape(true_w.shape)}")
print(f"b 的估计误差:{true_b - b}")


