import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)  # First hidden layer
        self.relu = nn.ReLU()                               # Activation function
        self.batch_norm = nn.BatchNorm1d(hidden_dim)        # Batch normalization
        self.dropout = nn.Dropout(0.5)                       # Dropout for regularization
        self.fc2 = nn.Linear(hidden_dim, vocab_size)         # Output layer
    
    def forward(self, x):
        x = x.view(x.size(0), -1)             # Ensure correct shape
        x = self.fc1(x)                       # Pass through first hidden layer
        x = self.relu(x)                      # Apply ReLU activation
        x = self.batch_norm(x)                # Apply batch normalization
        x = self.dropout(x)                   # Apply dropout
        x = self.fc2(x)                       # Pass through output layer
        return x

# 训练模型
def train_model(model, criterion, optimizer, x_train, y_train, embedding, num_epochs=1000):
    for epoch in range(num_epochs):
        model.train()
        inputs = torch.tensor([[char_to_idx[char] for char in pair] for pair in x_train], dtype=torch.long)
        targets = torch.tensor([char_to_idx[char] for char in y_train], dtype=torch.long)  # Targets as class indices
        embedded_inputs = embedding(inputs)

        optimizer.zero_grad()
        outputs = model(embedded_inputs)
        loss = criterion(outputs, targets)  # CrossEntropyLoss expects raw scores and class indices
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

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
    x_train, y_train = generate_data()

    # 初始化模型、损失函数和优化器
    hidden_dim = 64  # Increased hidden dimension
    model = AddNet(vocab_size, embedding_dim, hidden_dim)  # Updated model initialization
    criterion = nn.CrossEntropyLoss()                     # Ensure using CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=0.001)   # Optionally adjust learning rate

    # 训练模型
    train_model(model, criterion, optimizer, x_train, y_train, embedding)

    # 测试模型
    # Original single test case
    # test_expr = "一+七"
    # result = test_model(model, test_expr)
    # print(f'{test_expr} = {result}')
    
    # Added multiple test cases
    test_expressions = ["一+七", "二+三", "四+五", "六+零", "七+二", "三+四"]
    for expr in test_expressions:
        char_a, _, char_b = expr.partition('+')  # Split the expression into operands
        result = test_model(model, char_a, char_b, embedding)  # Provide all required arguments
        print(f'{expr} = {result}')
