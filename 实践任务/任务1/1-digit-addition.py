import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

# 生成数据集
def generate_data(num_samples=1000):
    x = []
    y = []
    while len(x) < num_samples:
        a, b = np.random.randint(0, 10, 2)
        sum_ab = a + b
        if sum_ab < 10:  # Ensure the sum is a single digit
            x.append([a, b])
            y.append(sum_ab)
    return np.array(x), np.array(y)

# 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 训练模型
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=200):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # 验证模型
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.4f}')

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
    
    # 转换为Tensor
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    # 创建DataLoader
    dataset = TensorDataset(x, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 初始化模型、损失函数和优化器
    input_dim = 2  # 输入维度为2，因为每个输入是两个数字
    hidden_dim = 64  # 隐藏层维度
    num_classes = 10  # 输出类别数
    model = SimpleNN(input_dim, hidden_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 学习率
    
    # 训练模型
    train_model(model, criterion, optimizer, train_loader, val_loader)
    
    # 测试模型
    test_model(model, val_loader)
