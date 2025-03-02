import torch
import torch.nn as nn

# 超参数
d_model = 512  # 模型的维度
nhead = 8      # 多头注意力头数
num_layers = 6 # 编码器和解码器的层数

# 输入序列长度、批量大小、词汇表大小
src_seq_len = 10
tgt_seq_len = 20
batch_size = 32
vocab_size = 5000

# 输入
src = torch.randint(0, vocab_size, (src_seq_len, batch_size))  # [seq_len, batch_size]
tgt = torch.randint(0, vocab_size, (tgt_seq_len, batch_size))  # [seq_len, batch_size]

# 嵌入层
embedding = nn.Embedding(vocab_size, d_model)

# Transformer 模型
encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)

# Encoder 和 Decoder 堆叠
encoder = nn.TransformerEncoder(encoder_layer, num_layers)
decoder = nn.TransformerDecoder(decoder_layer, num_layers)

# 通过嵌入层获得输入的表示
src_emb = embedding(src)
tgt_emb = embedding(tgt)

# 需要提供的 mask
src_mask = None
tgt_mask = None
memory_mask = None

# Transformer 编码器和解码器的前向传播
memory = encoder(src_emb)  # 编码器输出
output = decoder(tgt_emb, memory)  # 解码器输出

# 最终输出
print(output.shape)  # [tgt_seq_len, batch_size, d_model]
