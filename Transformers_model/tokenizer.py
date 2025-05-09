from tokenizers import Tokenizer, decoders, models, trainers, pre_tokenizers, processors

def build_bpe_tokenizer():
    """论文3.4节BPE分词器实现"""
    tokenizer = Tokenizer(models.BPE(
        unk_token="<unk>",
        fuse_unk=True  # 论文使用的unk处理方式
    ))
    
    # 配置预处理（论文5.2节）
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=True,
        use_regex=False  # 保持与论文一致
    )
    
    # 特殊符号与论文一致
    trainer = trainers.BpeTrainer(
        vocab_size=32000,
        special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    # 训练数据路径（请确认存在）
    tokenizer.train(files=[
        'data/train.src',
        'data/train.tgt'
    ], trainer=trainer)
    
    # 后处理配置（论文中的序列格式）
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A <eos>",
        pair="$A <eos> $B <eos>",
        special_tokens=[
            ("<eos>", tokenizer.token_to_id("<eos>"))
        ]
    )
    
    # configure byte-level decoder for proper text reconstruction
    tokenizer.decoder = decoders.ByteLevel()
    
    tokenizer.save("bpe_tokenizer.json")

if __name__ == "__main__":
    build_bpe_tokenizer()