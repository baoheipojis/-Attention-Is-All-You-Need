# encoder.py
import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff):
        super(EncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src, src_mask=None):
        attn_output, _ = self.attention(src, src, src, attn_mask=src_mask)
        src = self.norm1(src + self.dropout(attn_output))
        ffn_output = self.ffn(src)
        return self.norm2(src + self.dropout(ffn_output))
