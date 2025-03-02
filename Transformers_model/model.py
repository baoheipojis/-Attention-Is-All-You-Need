# model.py
import torch
import torch.nn as nn
import math
from encoder import EncoderLayer
from decoder import DecoderLayer
from positional_encoding import PositionalEncoding
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, d_ff, vocab_size, max_len=512):
        super(Transformer, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # 通过 Encoder 层堆叠
        self.encoder = nn.ModuleList([EncoderLayer(d_model, nhead, d_ff) for _ in range(num_encoder_layers)])
        
        # 通过 Decoder 层堆叠
        self.decoder = nn.ModuleList([DecoderLayer(d_model, nhead, d_ff) for _ in range(num_decoder_layers)])
        
        # 输出层
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # 这里的src的大小是(batch_size, sequence_length, d_model)
        # 论文3.4节的描述，先乘上d_model的平方根。
        src = self.embedding(src) * math.sqrt(src.size(2))
        tgt = self.embedding(tgt) * math.sqrt(tgt.size(2))
        
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        
        memory = src
        for encoder_layer in self.encoder:
            memory = encoder_layer(memory, src_mask)
        
        output = tgt
        for decoder_layer in self.decoder:
            output = decoder_layer(output, memory, tgt_mask, memory_mask)
        
        output = self.fc_out(output)
        return F.softmax(output, dim=-1)