import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# 定义词汇表
digits = [str(i) for i in range(10)]
operators = ['+', '-']
vocab = digits + operators
vocab_size = len(vocab)
char_to_idx = {ch: idx for idx, ch in enumerate(vocab)}
idx_to_char = {idx: ch for idx, ch in enumerate(vocab)}

# 生成数据集
def generate_data(num_samples=1000):
    expressions = []
    targets = []
    for _ in range(num_samples):
        num_terms = torch.randint(2, 5, (1,)).item()  # 表达式中数字的个数
        expr = digits[torch.randint(0, 10, (1,)).item()]  # 以数字开始
        for _ in range(num_terms - 1):
            expr += operators[torch.randint(0, 2, (1,)).item()]  # 添加操作符
            expr += digits[torch.randint(0, 10, (1,)).item()]    # 添加数字
        expressions.append(expr)
        targets.append(eval(expr))
    return expressions, targets

# 定义模型
class AddSubNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(AddSubNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        embeds = self.embedding(x)
        _, (hidden, _) = self.rnn(embeds)
        out = self.fc(hidden[-1])
        return out.squeeze()

# 编码表达式
def encode_expression(expr):
    return torch.tensor([char_to_idx[ch] for ch in expr], dtype=torch.long)

# 训练模型
def train_model(model, criterion, optimizer, x_train, y_train, x_val, y_val, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for expr, target in zip(x_train, y_train):
            optimizer.zero_grad()
            input_seq = encode_expression(expr).unsqueeze(0)
            output = model(input_seq)
            loss = criterion(output, torch.tensor(target, dtype=torch.float))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for expr, target in zip(x_val, y_val):
                input_seq = encode_expression(expr).unsqueeze(0)
                output = model(input_seq)
                loss = criterion(output, torch.tensor(target, dtype=torch.float))
                val_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(x_train):.4f}, Val Loss: {val_loss/len(x_val):.4f}')

# 测试模型
def test_model(model, expression):
    model.eval()
    with torch.no_grad():
        input_seq = encode_expression(expression).unsqueeze(0)
        output = model(input_seq)
    return output.item()

# 主函数
if __name__ == "__main__":
    # 生成数据
    expressions, targets = generate_data()
    x_train, x_val, y_train, y_val = train_test_split(expressions, targets, test_size=0.2)

    # 初始化模型、损失函数和优化器
    embedding_dim = 16
    hidden_dim = 32
    model = AddSubNet(vocab_size, embedding_dim, hidden_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, criterion, optimizer, x_train, y_train, x_val, y_val)

    # 测试模型
    test_expr = "1+2-2"
    result = test_model(model, test_expr)
    print(f'{test_expr} = {result:.2f}')