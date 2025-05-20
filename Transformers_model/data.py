from datasets import load_dataset

# 加载 WMT17 中文-英文翻译数据集
dataset = load_dataset("wmt19", "zh-en", cache_dir=".")

# 查看训练集中的前几个样本
for example in dataset['train'].select(range(10)):
    print(f"中文: {example['translation']['zh']}")
    print(f"英文: {example['translation']['en']}")
    print()