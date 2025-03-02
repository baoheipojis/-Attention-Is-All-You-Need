from collections import Counter
import torch
import torch.nn as nn

# 假设我们有以下训练数据
train_data = [
    "hello world",
    "this is an example",
    "attention is all you need"
]

# 预处理数据并构建词汇表
def build_vocab(data):
    counter = Counter()
    for sentence in data:
        words = sentence.split()
        counter.update(words)
    word_to_index = {word: idx for idx, (word, _) in enumerate(counter.items())}
    return word_to_index

# 构建词汇表
word_to_index = build_vocab(train_data)
vocab_size = len(word_to_index)
embedding_dim = 3  # 每个单词将被嵌入到一个3维向量中

# 创建一个Embedding层
embedding = nn.Embedding(vocab_size, embedding_dim)

# 将句子转换为索引
def sentence_to_indices(sentence, word_to_index):
    return torch.tensor([word_to_index[word] for word in sentence.split()], dtype=torch.long)

# 假设我们有一个句子 "this is an example"
sentence = "this is an example"
sentence_indices = sentence_to_indices(sentence, word_to_index)

# 使用Embedding层将单词索引转换为嵌入向量
embedded_sentence = embedding(sentence_indices)

print("词汇表:", word_to_index)
print("单词:", sentence.split())
print("单词索引:", sentence_indices)
print("嵌入向量:\n", embedded_sentence)