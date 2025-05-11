from transformers import BertTokenizer

# 加载BERT分词器（支持中英文，建议使用中文BERT模型）
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

def sentence_to_token_ids(sentence):
    """
    输入句子，输出token序列（token ids）
    """
    return tokenizer.encode(sentence, add_special_tokens=True)

def token_ids_to_tokens(token_ids):
    """
    输入token序号列表，输出token字符串列表
    """
    return tokenizer.convert_ids_to_tokens(token_ids)

if __name__ == "__main__":
    # 示例：中英文句子
    sentences = [
        "非常谢谢，克里斯。的确非常荣幸 能有第二次站在这个台上的机会，我真是非常感激。",
    ]
    for sent in sentences:
        ids = sentence_to_token_ids(sent)
        print(f"Sentence: {sent}")
        print(f"Token IDs: {ids}")
        print(f"Tokens: {token_ids_to_tokens(ids)}")
        print("-" * 30)

    # 示例：给定token id输出token字符串
    example_ids = [101, 3031, 671, 678,800,  812, 2218, 1359, 1957, 1398, 2595, 2605,       749,   102]  # 101/102是[CLS]/[SEP]
    print(f"Token IDs: {example_ids}")
    print(f"Tokens: {token_ids_to_tokens(example_ids)}")