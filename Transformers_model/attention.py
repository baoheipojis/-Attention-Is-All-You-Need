import torch
import torch.nn as nn
import math
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        # 初始化参数，默认为512维，8个头
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        # 这里权重矩阵就是(512,512)了。
        self.wq = nn.Linear(hidden_dim, hidden_dim)
        self.wk = nn.Linear(hidden_dim, hidden_dim)
        self.wv = nn.Linear(hidden_dim, hidden_dim)
        self.wo = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        seq_len = q.size(1)
        # 投影
        Q = self.wq(q).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = self.wk(k).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # 修改这里
        V = self.wv(v).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # 最后变成了一个(b,nh,s,hd)的矩阵，把nh放到第二维，方便模拟处理多头。
        # 计算注意力分数。Q是(b,nh,s,hd)，K^T是(b,nh,hd,s)，乘起来是(b,nh,s,s)，和我们之前的分析一致。
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            # 把mask为0的位置的分数设为一个很小的值，-1e9。这样在softmax的时候就会被忽略掉。
            scores = scores.masked_fill(mask == 0, -1e9)
        # 在最后一个维度上做softmax
        attn = torch.softmax(scores, dim=-1)
        # V是(b,nh,s,hd)，attn是(b,nh,s,s)，乘起来是(b,nh,s,hd)
        output = torch.matmul(attn, V)
        
        # 现在我们要把output的维度从(b,8,s,64)变成(b,s,512)。
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        return self.wo(output)