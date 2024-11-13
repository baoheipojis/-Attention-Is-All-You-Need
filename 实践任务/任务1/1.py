import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 
# device = torch.device("cpu")
# 数据准备
def generate_data(num_samples=5):
    # X的形状是(num_samples, 2)，从0到9之间随机选择
    X = np.random.randint(0, 10, (num_samples, 2))
    # axis=1表示对每行进行求和，每行的大小是2（就是列数）。
    y = X.sum(axis=1)
    return X, y

# 将数据转换为 PyTorch 张量
X, y = generate_data()
X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

# 定义模型
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # fc是全连接层，它会连接输入的所有特征和输出的所有特征。全连接层=线性层
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        # relu是把负值变0，正值不变。可以删掉试试看。其实对于本次实验是无所谓的。
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# 初始化模型、损失函数和优化器
model = SimpleMLP().to(device)
criterion = nn.MSELoss().to(device)
# 使用Adam优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10000
for epoch in range(num_epochs):
    # 一些层的表现在训练和推理下可能不同，这里把它们设为训练模式。
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    # 根据梯度更新参数
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 评估模型
model.eval()
with torch.no_grad():
    test_X = torch.tensor([[3, 7], [5, 2], [9, 0]], dtype=torch.float32)
    test_y = torch.tensor([[10], [7], [9]], dtype=torch.float32)
    test_outputs = model(test_X)
    test_loss = criterion(test_outputs, test_y)
    print(f'Test Loss: {test_loss.item():.4f}')
    print('Predictions:')
    for i in range(len(test_X)):
        print(f'Input: {test_X[i].numpy()}, Predicted: {test_outputs[i].item():.2f}, Actual: {test_y[i].item()}')