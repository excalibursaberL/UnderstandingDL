import torch
from torch.utils import data
from torch import nn

torch.manual_seed(42)

def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def load_array(data_arrays, batch_size, is_train = True):
    """构造 PyTorch 数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

# 指定实际参数
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# 指定超参数
batch_size = 10
data_iter = load_array((features, labels), batch_size)
# print(next(iter(data_iter)))

# 定义模型，初始化模型参数
net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 损失值
loss = nn.HuberLoss()

# 优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        # 前向传播计算预测值并计算损失
        l = loss(net(X), y)
        trainer.zero_grad()
        # 反向传播
        l.backward()
        trainer.step()

    l = loss(net(features), labels)
    print(f"epoch: {epoch}, loss: {l:f}")

w = net[0].weight.data
b = net[0].bias.data
print(f'w 的估计误差:{true_w - w.reshape(w.shape)}')
print(f'b 的估计误差:{true_b - b}')
