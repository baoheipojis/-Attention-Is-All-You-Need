import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 生成训练数据：所有可能的0-9组合及其和的个位数
inputs = torch.tensor([(a, b) for a in range(10) for b in range(10)], dtype=torch.long)
labels = torch.tensor([(a + b) % 10 for a in range(10) for b in range(10)], dtype=torch.long)

# 划分训练集和验证集（分层抽样保证类别分布）
X_train, X_val, y_train, y_val = train_test_split(
    inputs.numpy(), 
    labels.numpy(),
    test_size=0.2,
    stratify=labels.numpy(),  # 保持类别比例
    random_state=42
)

# 转回PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 定义神经网络（保持原有结构）
class AdditionModel(nn.Module):
    def __init__(self, embedding_dim=8, hidden_dim=24):
        super().__init__()
        self.embedding = nn.Embedding(10, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim*1, hidden_dim)  # 输入维度修正
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 10)
        
    def forward(self, x):
        a_emb = self.embedding(x[:, 0])
        b_emb = self.embedding(x[:, 1])
        combined = a_emb + b_emb  # 向量相加体现加法特性
        out = self.relu(self.fc1(combined))
        return self.fc2(out)

# 初始化模型组件
model = AdditionModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环（包含验证监控）
best_val_acc = 0
for epoch in range(300):
    # 训练阶段
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # 验证阶段
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    
    val_acc = 100 * correct / len(val_dataset)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
    
    print(f"Epoch {epoch+1}: Val Acc {val_acc:.1f}%")

# 加载最佳模型
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# 打印测试过程
print("\n测试过程详情：")
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        # 遍历当前batch中的每个样本
        for i in range(inputs.shape[0]):
            a = inputs[i, 0].item()
            b = inputs[i, 1].item()
            pred = predicted[i].item()
            true = labels[i].item()
            
            # 高亮显示正确/错误状态
            status = "✓" if pred == true else "✗"
            color_code = "\033[92m" if pred == true else "\033[91m"
            reset_code = "\033[0m"
            
            print(f"输入: {a:2d} + {b:2d} | " 
                  f"预测: {pred:2d} | "
                  f"真实: {true:2d} | "
                  f"{color_code}{status}{reset_code}")

# 最终准确率计算（保持原有逻辑）
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\n最终验证准确率: {100 * correct / total:.1f}%")