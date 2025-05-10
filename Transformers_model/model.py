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
        
        # 1. 在嵌入前创建掩码
        # src_mask大小是[batch_size, 1, 1, src_len]。这是因为对于每个batch都不同，但是对于序列中每个位置都相同。为什么要变四维呢？因为后面要分头。
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # 基于原始输入创建掩码
        
        tgt_len = tgt.size(1)
        # 解码器需要两种mask：padding mask和look-ahead mask
        # 创建padding mask - 在嵌入前基于原始输入
        # 修正：创建形状为 [batch_size, 1, tgt_len, tgt_len] 的padding mask
        tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, tgt_len]

        
        device = src.device
        
        # 生成look-ahead mask。triu函数生成上三角矩阵，diagonal表示从主对角线（左上到右下）开始的偏移量，这里为1表示主对角线不保留，
        look_ahead_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=device), 
            diagonal=1
        ).bool()
        
        # 将look_ahead_mask调整为合适的维度以便与padding mask组合
        # look_ahead_mask需要形状为[1, 1, tgt_len, tgt_len]，因为跟batch无关，所以第0维无所谓。
        look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0)
        look_ahead_mask = ~look_ahead_mask  # 取反，因为我们需要屏蔽未来位置
        # 将padding mask和look-ahead mask组合 - 维度现在兼容了
        tgt_mask = tgt_padding_mask & look_ahead_mask
        
        # 然后再执行嵌入
        src = self.input_embedding(src) * self.scale
        tgt = self.output_embedding(tgt) * self.scale
        
        # 2. 添加位置编码
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        
        # 3. 编码器
        for layer in self.encoder:
            src = layer(src, src_mask)
        

        
        for layer in self.decoder:
            tgt = layer(tgt, src, src_mask, tgt_mask)
        
        # 5. 输出层
        output = self.fc_out(tgt)
        # 眼尖的读者会发现这里相比原论文，少了一层softmax。这是因为不需要显式添加了，我们后面会解释。
        return output


if __name__ == "__main__":
    # 创建一个小型Transformer模型用于测试
    model = Transformer(d_model=64, nhead=4, num_encoder_layers=2, 
                       num_decoder_layers=2, d_ff=128, vocab_size=1000, max_len=100)
    
    # 创建示例输入批次
    batch_size = 2
    src_len = 5
    tgt_len = 4
    
    # 随机生成源序列和目标序列
    src = torch.randint(1, 1000, (batch_size, src_len))
    # 添加一些padding
    src[0, -2:] = 0
    print(f"源序列: \n{src}")
    
    tgt = torch.randint(1, 1000, (batch_size, tgt_len))
    tgt[1, -2:] = 0
    print(f"目标序列: \n{tgt}")
    
    
    # 运行模型
    print("\n--- 模型运行 ---")
    model.eval()
    with torch.no_grad():
        output = model(src, tgt)
        print(f"模型输入形状: src={src.shape}, tgt={tgt.shape}")
        print(f"模型输出形状: {output.shape}")
        
        # 最高概率预测
        predictions = torch.argmax(output, dim=-1)
        print(f"预测结果形状: {predictions.shape}")
        print(f"预测结果: \n{predictions}")
    
    print("\n模型执行成功!")

