# model.py
import torch
import torch.nn as nn
import math
from encoder import EncoderLayer
from decoder import DecoderLayer
from positional_encoding import PositionalEncoding

class Transformer(nn.Module):
    """
    Transformer 模型类，实现基于 Encoder-Decoder 架构的翻译模型。

    Args:
        d_mode (int): 模型隐藏层维度，也就是每个 token 的嵌入向量维度，通常设为 512。
        nhead (int): 多头注意力机制中的头数，必须能被 d_mode 整除，例如 512/8 = 64，每个头负责处理子空间的信息。
        num_encoder_layers (int): 编码器（Encoder）的层数，原论文中通常为 6。
        num_decoder_layers (int): 解码器（Decoder）的层数，原论文中通常为 6。
        d_ff (int): 前馈神经网络（Feed-Forward Network）的隐藏层维度，通常比 d_mode 大很多（例如 2048）。
        vocab_size (int): 词汇表大小，表示可处理的唯一 token 数量。根据具体任务和分词策略，一般为 32000 或其他数值。
        max_len (int): 模型能处理的最大序列长度，用于位置编码，通常设为 512。
    """
    def __init__(self, d_mode=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, vocab_size=32000, max_len=512):
        super(Transformer, self).__init__()
        
        self.input_embedding = nn.Embedding(src_vocab_size, d_model)  # 编码器端
        self.output_embedding = nn.Embedding(tgt_vocab_size, d_model) # 解码器端
        # 这是论文3.4节提到的缩放因子
        self.scale = math.sqrt(d_model)        # added scale factor
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # 通过 Encoder 层堆叠
        self.encoder = nn.ModuleList([EncoderLayer(d_model, nhead, d_ff) for _ in range(num_encoder_layers)])
        
        # 通过 Decoder 层堆叠
        self.decoder = nn.ModuleList([DecoderLayer(d_model, nhead, d_ff) for _ in range(num_decoder_layers)])
        
        # 输出层
        self.fc_out = nn.Linear(d_model, vocab_size)

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
        
        # 3. 编码器
        for layer in self.encoder:
            src = layer(src)
        
        # 4. 解码器
        for layer in self.decoder:
          # 这里decoder需要接受encoder的输出，以及自己的输入，两部分。图上可以看到，encoder有两根线连到decoder，同时decoder还有自己的输入。
            tgt = layer(tgt, src)
        
        # 5. 输出层
        output = self.fc_out(tgt)
        # 眼尖的读者会发现这里相比原论文，少了一层softmax。这是因为不需要显式添加了，我们后面会解释。
        return output