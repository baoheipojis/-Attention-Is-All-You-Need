# model.py
import torch
import torch.nn as nn
import math
from encoder import EncoderLayer
from decoder import DecoderLayer
from positional_encoding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, 
                num_decoder_layers=6, d_ff=2048, vocab_size=32000, max_len=512,dropout=0.1):  # Added max_len
        super().__init__()
        
        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.output_embedding = nn.Embedding(vocab_size, d_model)
        self.scale = math.sqrt(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)  # Now using defined max_len
        
        # 通过 Encoder 层堆叠
        # 修改Encoder/Decoder初始化方式
        self.encoder = nn.ModuleList([EncoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, nhead, d_ff, dropout) for _ in range(num_decoder_layers)])
        
        # 输出层
        self.fc_out = nn.Linear(d_model, vocab_size)

    # 在forward函数中增加mask处理
    def forward(self, src, tgt):
        """
        前向传播函数。

        Args:
            src (Tensor): 输入序列，形状为 (batch_size, src_len)。
            tgt (Tensor): 目标序列，形状为 (batch_size, tgt_len)。

        Returns:
            Tensor: 模型输出，形状为 (batch_size, tgt_len, vocab_size)。
        """
        
        # 1. 嵌入输入
        src = self.input_embedding(src) * self.scale
        tgt = self.output_embedding(tgt) * self.scale
        
        # 2. 添加位置编码
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        # 修正src_mask生成逻辑
        src_mask = (src[:, :, 0] != 0).unsqueeze(1).unsqueeze(2)  # 取嵌入后的序列长度维度
          
        # 3. 编码器
        for layer in self.encoder:
            src = layer(src,src_mask)
        
        # 4. 解码器
        # 生成mask
        # 修改mask生成逻辑

        # 解码器需要两种mask
        # 修正padding mask维度
        tgt_padding_mask = (tgt[:, :, 0] != 0).unsqueeze(1).unsqueeze(3).bool()  # 添加.bool()
        seq_length = tgt.size(1)
        device = src.device
        
        # 生成look-ahead mask并调整维度
        look_ahead_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=device), 
            diagonal=1
        ).unsqueeze(0).unsqueeze(0).bool()  # 添加.bool()
        
        # 组合mask时应保持维度一致
        tgt_mask = tgt_padding_mask & look_ahead_mask
        
        for layer in self.decoder:
            tgt = layer(tgt, src, src_mask, tgt_mask)
        
        # 5. 输出层
        output = self.fc_out(tgt)
        # 眼尖的读者会发现这里相比原论文，少了一层softmax。这是因为不需要显式添加了，我们后面会解释。
        return output