import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cpu")

# 操作符列表
operators = ['+', '-']
digits = list("0123456789")

# 生成训练数据
def generate_data(num_samples=5000):  # Increased number of samples
    x = []
    y = []
    for _ in range(num_samples):
        length = np.random.randint(3, 10)  # 随机生成表达式长度
        expression = ""
        result = 0
        for i in range(length):
            if i % 2 == 0:
                num = np.random.randint(0, 10)
                expression += str(num)
                if i == 0:
                    result = num
                else:
                    if expression[i-1] == '+':
                        result += num
                    elif expression[i-1] == '-':
                        result -= num
            else:
                op = np.random.choice(operators)
                expression += op
        x.append([digits.index(char) if char in digits else len(digits) + operators.index(char) for char in expression])
        y.append(result)
    # Normalize targets
    y = np.array(y, dtype=np.float32)
    y = (y - y.mean()) / y.std()
    return x, y

# 定义神经网络模型
class AddSubNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2):  # Added num_layers
        super(AddSubNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        x = self.fc(hn[-1])
        return x

# 训练模型
def train_model(model, criterion, optimizer, x_train, y_train, num_epochs=2000):  # Increased epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)  # Added scheduler
    for epoch in range(num_epochs):
        model.train()
        inputs = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq, dtype=torch.long).to(device) for seq in x_train], batch_first=True)
        targets = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
def test_model(model, expression):
    model.eval()
    with torch.no_grad():
        tokens = [digits.index(char) if char in digits else len(digits) + operators.index(char) for char in expression]
        inputs = torch.tensor([tokens], dtype=torch.long).to(device)
        output = model(inputs)
        # Denormalize output
        output = output.item() * y_std + y_mean
        return output

# 主函数
if __name__ == "__main__":
    # 生成数据
    x_train, y_train = generate_data()

    # 计算均值和标准差 for normalization
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    y_train = (y_train - y_mean) / y_std

    # 初始化模型、损失函数和优化器
    vocab_size = len(digits) + len(operators)
    embedding_dim = 32  # Increased embedding dimension
    hidden_dim = 64  # Increased hidden dimension
    model = AddSubNet(vocab_size, embedding_dim, hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjusted learning rate

    # 训练模型
    train_model(model, criterion, optimizer, x_train, y_train)

    # 测试模型
    expression = "1+2+1-3"
    result = test_model(model, expression)
    print(f'{expression} = {result:.2f}')