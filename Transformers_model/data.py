from datasets import load_dataset

def load_data(split="train", sample_size=200000):
    """
    Load zh→en pairs from WMT19.
    split: one of "train","validation","test"
    sample_size: number of examples to load (default 10000).
    """
    # Use streaming to avoid downloading the entire dataset
    dataset = load_dataset("wmt19", cache_dir="./cache_small", streaming=True)
    ds = dataset[split]
    
    src_texts = []
    tgt_texts = []
    
    # Only process the required number of samples
    for i, ex in enumerate(ds):
        if sample_size and i >= sample_size:
            break
        src_texts.append(ex["translation"]["zh"])
        tgt_texts.append(ex["translation"]["en"])
    
    return src_texts, tgt_texts