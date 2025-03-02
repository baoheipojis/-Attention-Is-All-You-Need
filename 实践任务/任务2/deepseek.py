import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 生成数据集（符号使用独立编码）
def generate_data():
    add_samples = []
    for a in range(10):
        for b in range(10):
            if a + b <= 10:
                add_samples.append((a, 10, b, a + b))  # 加号编码为10
    
    sub_samples = []
    for a in range(10):
        for b in range(a + 1):
            sub_samples.append((a, 11, b, a - b))  # 减号编码为11
    
    return add_samples + sub_samples

# 自定义Dataset类
class UnifiedMathDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        a, op, b, result = self.data[idx]
        return (
            torch.tensor(a, dtype=torch.long),    # 数字0-9
            torch.tensor(op, dtype=torch.long),   # 符号10/11
            torch.tensor(b, dtype=torch.long),    # 数字0-9
            torch.tensor(result, dtype=torch.long)
        )

# 统一Embedding的神经网络模型
class UnifiedMathModel(nn.Module):
    def __init__(self, embed_dim=8):
        super().__init__()
        # 统一处理数字和符号的Embedding层
        self.embed = nn.Embedding(12, embed_dim)  # 0-9为数字，10-11为符号
        self.fc1 = nn.Linear(embed_dim*3, 32)
        self.fc2 = nn.Linear(32, 11)  # 输出0-10共11种可能
        
    def forward(self, a, op, b):
        # 统一进行Embedding处理
        a_emb = self.embed(a)
        op_emb = self.embed(op)
        b_emb = self.embed(b)
        combined = torch.cat([a_emb, op_emb, b_emb], dim=1)
        x = F.relu(self.fc1(combined))
        return self.fc2(x)

# 参数设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 100
EMBED_DIM = 8
LEARNING_RATE = 0.001

# 准备数据
full_data = generate_data()
train_data, test_data = train_test_split(full_data, test_size=0.2, random_state=42)

train_dataset = UnifiedMathDataset(train_data)
test_dataset = UnifiedMathDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 初始化模型
model = UnifiedMathModel(EMBED_DIM).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练循环（与之前相同）
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for a, op, b, labels in train_loader:
        a, op, b, labels = a.to(device), op.to(device), b.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(a, op, b)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

# 测试和预测（与之前相同）
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for a, op, b, labels in test_loader:
        a, op, b, labels = a.to(device), op.to(device), b.to(device), labels.to(device)
        outputs = model(a, op, b)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# 更新后的示例预测
test_cases = [
    (1, 10, 1),  # 1 + 1
    (3, 11, 2),  # 3 - 2
    (9, 10, 1),  # 9 + 1
    (5, 11, 3)   # 5 - 3
]

model.eval()
with torch.no_grad():
    for a, op, b in test_cases:
        a_t = torch.tensor([a], dtype=torch.long).to(device)
        op_t = torch.tensor([op], dtype=torch.long).to(device)
        b_t = torch.tensor([b], dtype=torch.long).to(device)
        output = model(a_t, op_t, b_t)
        prediction = torch.argmax(output).item()
        print(f"{a} {'+' if op ==10 else '-'} {b} = {prediction}")
import re

def predict_from_input():
    print("\n请输入数学表达式（例如 '1+1' 或 '3-2'），或输入 'quit' 退出。")
    while True:
        expr = input("表达式: ").strip()
        if expr.lower() == "quit":
            break

        # 使用正则表达式提取操作数和运算符
        match = re.match(r'(\d+)\s*([\+\-])\s*(\d+)', expr)
        if not match:
            print("无效表达式，请重试。")
            continue

        a_str, op, b_str = match.groups()
        a = int(a_str)
        b = int(b_str)

        # 根据模型训练时的编码，'+' 编码为 10，'-' 编码为 11
        op_token = 10 if op == '+' else 11

        a_t = torch.tensor([a], dtype=torch.long).to(device)
        op_t = torch.tensor([op_token], dtype=torch.long).to(device)
        b_t = torch.tensor([b], dtype=torch.long).to(device)
        
        with torch.no_grad():
            output = model(a_t, op_t, b_t)
            prediction = torch.argmax(output).item()
        
        print(f"{a} {op} {b} = {prediction}")

# 调用交互式预测函数
predict_from_input()