# decoder.py
import torch
import torch.nn as nn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff):
        super(DecoderLayer, self).__init__()
        self.masked_attention = nn.MultiheadAttention(d_model, nhead)
        self.encoder_decoder_attention = nn.MultiheadAttention(d_model, nhead)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        masked_attention_output, _ = self.masked_attention(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout(masked_attention_output))
        encoder_decoder_output, _ = self.encoder_decoder_attention(tgt, memory, memory, attn_mask=memory_mask)
        tgt = self.norm2(tgt + self.dropout(encoder_decoder_output))
        ffn_output = self.ffn(tgt)
        return self.norm3(tgt + self.dropout(ffn_output))
