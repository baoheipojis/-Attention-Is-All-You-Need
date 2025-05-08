# decoder.py
import torch
import torch.nn as nn
from attention import MultiHeadAttention
from ffn import PositionwiseFeedForward
class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        # decoder有两层attention，一个是self attention，一个是cross attention。cross attention的接受encoder的输出。
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads)
        self.cross_attn = MultiHeadAttention(hidden_dim, num_heads)
        self.ffn = PositionwiseFeedForward(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        return x