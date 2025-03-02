import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# 定义数据集：样本为 (x, y, op) ，其中 op: 0 表示加法，1 表示减法
# 对于加法：标签为 x+y  (范围 0~18)
# 对于减法：标签为 (x-y)+9  (将 -9~9 映射到 0~18)
class CalcDataset(Dataset):
    def __init__(self):
        self.samples = []
        # 加法
        for x in range(10):
            for y in range(10):
                self.samples.append((x, y, 0, x+y))
        # 减法
        for x in range(10):
            for y in range(10):
                self.samples.append((x, y, 1, (x-y) + 9))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y, op, label = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), \
               torch.tensor(y, dtype=torch.long), \
               torch.tensor(op, dtype=torch.long), \
               torch.tensor(label, dtype=torch.long)

# 定义神经网络模型，使用 Embedding 对三个输入进行编码，再将它们拼接后通过全连接层输出分类结果
class CalcNet(nn.Module):
    def __init__(self, emb_dim=10, hidden_dim=16, num_classes=19):
        super(CalcNet, self).__init__()
        self.emb_digits = nn.Embedding(10, emb_dim)
        self.emb_op = nn.Embedding(2, emb_dim)
        self.fc1 = nn.Linear(3 * emb_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, y, op):
        x_emb = self.emb_digits(x)
        y_emb = self.emb_digits(y)
        op_emb = self.emb_op(op)
        vec = torch.cat([x_emb, y_emb, op_emb], dim=-1)
        out = self.fc1(vec)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 参数初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CalcNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 加载数据集并划分训练集和测试集
dataset = CalcDataset()
total_samples = len(dataset)
train_size = int(0.8 * total_samples)
test_size = total_samples - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for x, y, op, label in train_loader:
        x, y, op, label = x.to(device), y.to(device), op.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = model(x, y, op)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(train_loader):.4f}")

# 在测试集上进行评估
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y, op, label in test_loader:
        x, y, op, label = x.to(device), y.to(device), op.to(device), label.to(device)
        outputs = model(x, y, op)
        _, predicted = torch.max(outputs, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
print(f"Test Accuracy: {100 * correct / total:.2f}%")
# Detailed evaluation: print expression, predicted result, and true result
print("Detailed Evaluation on Test Set:")
model.eval()
with torch.no_grad():
    for x, y, op, label in test_loader:
        x, y, op, label = x.to(device), y.to(device), op.to(device), label.to(device)
        outputs = model(x, y, op)
        _, predicted = torch.max(outputs, 1)
        for i in range(len(label)):
            x_val = x[i].item()
            y_val = y[i].item()
            op_val = op[i].item()  # 0 for addition, 1 for subtraction
            # Compute the correct result for printing
            true_result = x_val + y_val if op_val == 0 else x_val - y_val
            # Adjust prediction: for subtraction, subtract 9 to convert the output label back
            pred = predicted[i].item()
            pred_result = pred if op_val == 0 else pred - 9
            symbol = '+' if op_val == 0 else '-'
            print(f"Expression: {x_val} {symbol} {y_val}, Predicted: {pred_result}, True: {true_result}")
# Detailed Evaluation on Train Set (first 10 samples)
print("Detailed Evaluation on Train Set (first 10 samples):")
model.eval()
with torch.no_grad():
    count = 0
    for x, y, op, label in train_loader:
        x, y, op, label = x.to(device), y.to(device), op.to(device), label.to(device)
        outputs = model(x, y, op)
        _, predicted = torch.max(outputs, 1)
        for i in range(len(label)):
            x_val = x[i].item()
            y_val = y[i].item()
            op_val = op[i].item()  # 0 for addition, 1 for subtraction
            # Compute the correct result for printing
            true_result = x_val + y_val if op_val == 0 else x_val - y_val
            # Adjust prediction: for subtraction, subtract 9 to convert output label back
            pred = predicted[i].item()
            pred_result = pred if op_val == 0 else pred - 9
            symbol = '+' if op_val == 0 else '-'
            print(f"Expression: {x_val} {symbol} {y_val}, Predicted: {pred_result}, True: {true_result}")
            count += 1
            if count >= 10:
                break
        if count >= 10:
            break
# 测试预测函数
def predict(x, y, op):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor([x], dtype=torch.long).to(device)
        y_tensor = torch.tensor([y], dtype=torch.long).to(device)
        op_tensor = torch.tensor([op], dtype=torch.long).to(device)
        output = model(x_tensor, y_tensor, op_tensor)
        pred = torch.argmax(output, dim=1).item()
        return pred if op == 0 else pred - 9

print("测试加法：7 + 8 =", predict(7, 8, 0))
print("测试减法：7 - 8 =", predict(7, 8, 1))