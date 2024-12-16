import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 汉字列表
chinese_chars = list("零一二三四五六七八九")

# 定义embedding层
embedding_dim = 10  # 嵌入向量的维度
vocab_size = len(chinese_chars)
embedding = nn.Embedding(vocab_size, embedding_dim)

# 将汉字转换为索引
char_to_idx = {ch: idx for idx, ch in enumerate(chinese_chars)}
idx_to_char = {idx: ch for idx, ch in enumerate(chinese_chars)}

# 示例汉字
chars = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]

# 将汉字转换为索引
indices = torch.tensor([char_to_idx[ch] for ch in chars], dtype=torch.long)

# 获取嵌入向量
embedded_chars = embedding(indices)

print("汉字:", chars)
print("嵌入向量:\n", embedded_chars)
