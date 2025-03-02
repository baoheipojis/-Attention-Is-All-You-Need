# positional_encoding.py

# 论文3.5节中提到了位置编码
import torch
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        # 生成一个形状为(max_len, 1)的张量，其中包含从0到max_len-1的值
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term就是分母10000^(2i/d_model)倒数，这里i是维度索引。因为embedding的维度是d_model，每个维度上都要有位置编码，不同维度不同，i就是维度。例如第0维的位置编码是10000^(2*0/d_model)。
        # 这个地方是Postional Encoding中最复杂的地方了，其实也没多复杂，就是想办法把让底数等于10000。因为exp(log(10000))=10000，那么exp(log(10000) * x) = 10000^x，就很容易了。这个式子的结果是10000^(-2i/d_model)。
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # 这是一个切片操作，第一个:表示选择所有行，第二个0::2表示从第0列开始，每隔2列选择一次。
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 把pe的大小变成(1, max_len, d_model)
        pe = pe.unsqueeze(0)
        # 使pe成为模型的一部分，可以随设备转换。
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 假如x的大小是(batch_size, seq_len, d_model)，由于pe的大小是(1, max_len, d_model)，只要让pe的第一个维度和x一样，就可以直接加了。
        return x + self.pe[:, :x.size(1)]
