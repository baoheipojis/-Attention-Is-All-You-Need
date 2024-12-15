import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 汉字列表
chinese_chars = list("零一二三四五六七八九")

# 生成训练数据
def generate_data(num_samples=50):
    x = np.random.choice(chinese_chars, (num_samples, 2))
    y = np.array([chinese_chars.index(a) + chinese_chars.index(b) for a, b in x])
    return x, y

# 定义神经网络模型
class AddNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(AddNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)  # 展平嵌入向量
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 训练模型
def train_model(model, criterion, optimizer, x_train, y_train, num_epochs=1000):
    for epoch in range(num_epochs):
        model.train()
        inputs = torch.tensor([[chinese_chars.index(char) for char in pair] for pair in x_train], dtype=torch.long)
        targets = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
def test_model(model, char_a, char_b):
    model.eval()
    with torch.no_grad():
        idx_a = chinese_chars.index(char_a)
        idx_b = chinese_chars.index(char_b)
        inputs = torch.tensor([[idx_a, idx_b]], dtype=torch.long)
        output = model(inputs)
        return output.item()

# 主函数
if __name__ == "__main__":
    # 生成数据
    x_train, y_train = generate_data()

    # 初始化模型、损失函数和优化器
    vocab_size = len(chinese_chars)  # 输入汉字的范围
    embedding_dim = 10  # 嵌入向量的维度
    model = AddNet(vocab_size, embedding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练模型
    train_model(model, criterion, optimizer, x_train, y_train)

    # 测试模型
    char_a, char_b = '一', '二'
    c = test_model(model, char_a, char_b)
    print(f'{char_a} + {char_b} = {c:.2f}')

    # 进一步测试 a + d = e
    char_d = '三'
    e = test_model(model, char_a, char_d)
    print(f'{char_a} + {char_d} = {e:.2f}')