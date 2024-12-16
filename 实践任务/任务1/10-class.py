import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np  # Corrected import statement

# 生成数据集
def generate_data(num_samples=1000):
    x = np.random.rand(num_samples, 20, 1)  # 生成随机特征，增加一个维度以适应RNN输入
    y = np.random.randint(0, 10, num_samples)  # 生成0到9的随机标签
    return x, y

# 定义RNN模型
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        c0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

# 训练模型
def train_model(model, criterion, optimizer, scheduler, dataloader, num_epochs=300):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# 测试模型
def test_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')

# 主函数
if __name__ == "__main__":
    # 生成数据
    x, y = generate_data()
    
    # 数据归一化
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    
    # 转换为Tensor
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    # 创建DataLoader
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 初始化模型、损失函数和优化器
    input_dim = 1  # 输入维度为1，因为每个特征是一个标量
    hidden_dim = 128  # 隐藏层维度
    num_layers = 2  # RNN层数
    num_classes = 10
    model = RNNModel(input_dim, hidden_dim, num_layers, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 学习率
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # 学习率调度器
    
    # 训练模型
    train_model(model, criterion, optimizer, scheduler, dataloader)
    
    # 测试模型
    test_model(model, dataloader)
