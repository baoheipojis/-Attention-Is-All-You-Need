import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

# 汉字列表
chinese_chars = list("零一二三四五六七八九")
vocab_size = len(chinese_chars)
char_to_idx = {ch: idx for idx, ch in enumerate(chinese_chars)}
idx_to_char = {idx: ch for idx, ch in enumerate(chinese_chars)}

# 测试样例
test_expressions = ["一+七", "二+三", "四+五", "六+零", "七+二", "三+四"]

# 生成数据集
def generate_data(num_samples=100):
    expressions = []
    targets = []
    while len(expressions) < num_samples:
        a, b = np.random.choice(chinese_chars, 2)
        sum_index = chinese_chars.index(a) + chinese_chars.index(b)
        expr = a + "+" + b
        if sum_index < len(chinese_chars) and expr not in test_expressions:
            expressions.append(expr)
            targets.append(chinese_chars[sum_index])
    return expressions, targets

# 定义模型
class AddNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(AddNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embeds = self.embedding(x)
        _, (hidden, _) = self.rnn(embeds)
        out = self.fc(hidden[-1])
        return out.squeeze()

# 编码表达式
def encode_expression(expr):
    return torch.tensor([char_to_idx[ch] for ch in expr if ch != '+'], dtype=torch.long)

# 训练模型
def train_model(model, criterion, optimizer, expressions, targets, epochs=30):
    model.train()
    for epoch in range(epochs):
        # 重新划分训练集和验证集
        x_train, x_val, y_train, y_val = train_test_split(expressions, targets, test_size=0.2)

        total_loss = 0
        for expr, target in zip(x_train, y_train):
            optimizer.zero_grad()
            input_seq = encode_expression(expr).unsqueeze(0)
            output = model(input_seq).unsqueeze(0)  # 添加一个维度
            loss = criterion(output, torch.tensor([char_to_idx[target]], dtype=torch.long))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for expr, target in zip(x_val, y_val):
                input_seq = encode_expression(expr).unsqueeze(0)
                output = model(input_seq).unsqueeze(0)  # 添加一个维度
                loss = criterion(output, torch.tensor([char_to_idx[target]], dtype=torch.long))
                val_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(x_train):.4f}, Val Loss: {val_loss/len(x_val):.4f}')

# 测试模型
def test_model(model, expression):
    model.eval()
    with torch.no_grad():
        input_seq = encode_expression(expression).unsqueeze(0)
        output = model(input_seq)
        predicted_index = torch.argmax(output, dim=0).item()  # 修改为 dim=0
    return idx_to_char[predicted_index]

# 主函数
if __name__ == "__main__":
    # 生成数据
    expressions, targets = generate_data()

    # 初始化模型、损失函数和优化器
    embedding_dim = 16
    hidden_dim = 32
    model = AddNet(vocab_size, embedding_dim, hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, criterion, optimizer, expressions, targets)

    # 测试模型
    for expr in test_expressions:
        result = test_model(model, expr)
        print(f'{expr} = {result}')
