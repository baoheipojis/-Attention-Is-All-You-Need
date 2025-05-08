# encoder.py
import torch
import torch.nn as nn
from attention import MultiHeadAttention
from ffn import PositionwiseFeedForward
class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads)
        self.ffn = PositionwiseFeedForward(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        # 为什么2个norm，一个dropout呢。这是因为norm是有可学习参数的，两层norm需要区分开，但是dropout就无所谓了，即使是同一个，每次调用的结果也是不一样的。
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 这里的attn怎么qkv输入都是x呢？这是因为，可学参数在权重W^Q,W^K,W^V上面，看结构图你也确实可以发现，multi-head attention这个层是由一个输入分成三份输入的。那为什么要用3个x当参数呢？既然都一样就直接用一个x，在attn里用3次不就行了？这是因为一会在decoder里会不一样的，别急。
        attn_output = self.self_attn(x, x, x, mask)
        # 这里是5.4提到的，应用dropout和残差连接
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x