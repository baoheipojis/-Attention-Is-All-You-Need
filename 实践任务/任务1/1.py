import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

# 汉字列表
chinese_chars = list("零一二三四五六七八九")

# 定义embedding层
embedding_dim = 10  # 嵌入向量的维度
vocab_size = len(chinese_chars)
embedding = nn.Embedding(vocab_size, embedding_dim)

# 将汉字转换为索引
char_to_idx = {ch: idx for idx, ch in enumerate(chinese_chars)}
idx_to_char = {idx: ch for idx, ch in enumerate(chinese_chars)}

# 示例汉字
chars = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]

# 将汉字转换为索引
indices = torch.tensor([char_to_idx[ch] for ch in chars], dtype=torch.long)

# 获取嵌入向量
embedded_chars = embedding(indices)

print("汉字:", chars)
print("嵌入向量:\n", embedded_chars)

# 生成训练数据
def generate_data(num_samples=100):
    x = []
    y = []
    while len(x) < num_samples:
        a, b = np.random.choice(chinese_chars, 2)
        sum_index = chinese_chars.index(a) + chinese_chars.index(b)
        if sum_index < len(chinese_chars):
            x.append([a, b])
            y.append(chinese_chars[sum_index])
    return np.array(x), np.array(y)

# 定义神经网络模型
class AddNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(AddNet, self).__init__()
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 计算准确率
def calculate_accuracy(model, x_val, y_val, embedding):
    model.eval()
    correct = 0
    total = len(y_val)
    with torch.no_grad():
        val_inputs = torch.tensor([[char_to_idx[char] for char in pair] for pair in x_val], dtype=torch.long)
        val_targets = torch.tensor([char_to_idx[char] for char in y_val], dtype=torch.long)
        embedded_val_inputs = embedding(val_inputs)
        val_outputs = model(embedded_val_inputs)
        _, predicted = torch.max(val_outputs, 1)
        correct = (predicted == val_targets).sum().item()
    accuracy = correct / total
    return accuracy

# 训练模型
def train_model(model, criterion, optimizer, x_train, y_train, x_val, y_val, embedding, num_epochs=3000, patience=10):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        inputs = torch.tensor([[char_to_idx[char] for char in pair] for pair in x_train], dtype=torch.long)
        targets = torch.tensor([char_to_idx[char] for char in y_train], dtype=torch.long)
        embedded_inputs = embedding(inputs)

        optimizer.zero_grad()
        outputs = model(embedded_inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            val_inputs = torch.tensor([[char_to_idx[char] for char in pair] for pair in x_val], dtype=torch.long)
            val_targets = torch.tensor([char_to_idx[char] for char in y_val], dtype=torch.long)
            embedded_val_inputs = embedding(val_inputs)
            val_outputs = model(embedded_val_inputs)
            val_loss = criterion(val_outputs, val_targets)
            accuracy = calculate_accuracy(model, x_val, y_val, embedding)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Accuracy: {accuracy:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

# 测试模型
def test_model(model, char_a, char_b, embedding):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor([[char_to_idx[char_a], char_to_idx[char_b]]], dtype=torch.long)
        embedded_inputs = embedding(inputs)
        output = model(embedded_inputs)
        predicted_index = torch.argmax(output, dim=1).item()
        return idx_to_char[predicted_index]

# 主函数
if __name__ == "__main__":
    # 生成数据
    x, y = generate_data()
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    # 初始化模型、损失函数和优化器
    hidden_dim = 16  # Further reduced hidden dimension
    model = AddNet(vocab_size, embedding_dim, hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)  # Reduced learning rate

    # 训练模型
    train_model(model, criterion, optimizer, x_train, y_train, x_val, y_val, embedding)

    # 测试模型
    test_expressions = ["一+七", "二+三", "四+五", "六+零", "七+二", "三+四"]
    for expr in test_expressions:
        char_a, _, char_b = expr.partition('+')
        result = test_model(model, char_a, char_b, embedding)
        print(f'{expr} = {result}')
