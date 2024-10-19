import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 数据准备
def generate_data(num_samples=5):
    X = np.random.randint(0, 10, (num_samples, 2))
    y = X.sum(axis=1)
    return X, y

# 将数据转换为 PyTorch 张量
X, y = generate_data()
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# 定义模型
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化模型、损失函数和优化器
model = SimpleMLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
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