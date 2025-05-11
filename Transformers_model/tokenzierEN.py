from transformers import BertTokenizer

# 加载英文BERT分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def sentence_to_token_ids(sentence):
    return tokenizer.encode(sentence, add_special_tokens=True)

def token_ids_to_tokens(token_ids):
    return tokenizer.convert_ids_to_tokens(token_ids)

if __name__ == "__main__":
    sentences = [
        "Thank you very much, Chris. It's truly an honor to have this opportunity to speak here again.",
    ]
    
    for sent in sentences:
        ids = sentence_to_token_ids(sent)
        print(f"Sentence: {sent}")
        print(f"Token IDs: {ids}")
        print(f"Tokens: {token_ids_to_tokens(ids)}")
        print("-" * 30)

    # 示例token id转换
    example_ids = [101,  6073,  2009,  1010,  1998,  2017,  2031,  1037, 11690,  3232,
          1012,   102   ]
    print(f"Token IDs: {example_ids}")
    print(f"Tokens: {token_ids_to_tokens(example_ids)}")