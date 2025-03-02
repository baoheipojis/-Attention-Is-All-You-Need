import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

# 生成数据集
def generate_data(num_samples=5000, exclude=None):
    x = []
    y = []
    exclude = exclude or []
    exclude_set = set(tuple(e) for e in exclude)
    exclude = exclude or []
    exclude_set = set(tuple(e) for e in exclude)
    while len(x) < num_samples:
        a, b = np.random.randint(0, 10, 2)
        if (a, b) not in exclude_set and (b, a) not in exclude_set:
            sum_ab = a + b
            if sum_ab < 10:  # Ensure the sum is a single digit
                x.append([a, b])
                y.append(sum_ab)
        if (a, b) not in exclude_set and (b, a) not in exclude_set:
            sum_ab = a + b
            if sum_ab < 10:  # Ensure the sum is a single digit
                x.append([a, b])
                y.append(sum_ab)
    return np.array(x), np.array(y)

# 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_classes):
        super(SimpleNN, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x.long())
        x = x.view(x.size(0), -1)  # Flatten the embedding vectors
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

# 训练模型
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=100):
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=100):
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
            print(f'Input: {inputs.numpy()}, Predicted: {predicted.item()}, Actual: {targets.item()}')
            print(f'Input: {inputs.numpy()}, Predicted: {predicted.item()}, Actual: {targets.item()}')
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')

# 主函数
if __name__ == "__main__":
    # 生成测试数据
    test_data = np.array([[1, 2], [3, 4], [5, 4], [7, 2], [9, 0]])
    test_labels = np.array([3, 7, 9, 9, 9])
    
    # 生成训练数据，排除测试数据
    x, y = generate_data(exclude=test_data.tolist())
    # 生成测试数据
    test_data = np.array([[1, 2], [3, 4], [5, 4], [7, 2], [9, 0]])
    test_labels = np.array([3, 7, 9, 9, 9])
    
    # 生成训练数据，排除测试数据
    x, y = generate_data(exclude=test_data.tolist())
    
    # 转换为Tensor
    x = torch.tensor(x, dtype=torch.long)  # 使用long类型以便嵌入层处理
    y = torch.tensor(y, dtype=torch.long)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    
    # 创建DataLoader
    dataset = TensorDataset(x, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=1, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=1, shuffle=False)
    
    # 初始化模型、损失函数和优化器
    input_dim = 2  # 输入维度为2，因为每个输入是两个数字
    hidden_dim = 128  # 隐藏层维度
    num_classes = 10  # 输出类别数
    model = SimpleNN(input_dim, embedding_dim, hidden_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 学习率
    
    # 训练模型
    train_model(model, criterion, optimizer, train_loader, val_loader)
    
    # 测试模型
    test_model(model, test_loader)
    test_model(model, test_loader)
