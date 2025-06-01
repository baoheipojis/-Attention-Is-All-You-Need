import torch
from transformers import BertTokenizer
import sacrebleu
import os

from transformer_translation import (
    TransformerTranslator,
    translate,
    load_data,
    BEST_MODEL_PATH,
    FINAL_MODEL_PATH,
    D_MODEL,
    N_HEAD,
    NUM_ENCODER_LAYERS,
    NUM_DECODER_LAYERS,
    DIM_FEEDFORWARD,
    DROPOUT,
    MAX_LEN,
    USE_CUSTOM_TRANSFORMER,
)
FINAL_MODEL_PATH = BEST_MODEL_PATH

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    model_type = "custom" if USE_CUSTOM_TRANSFORMER else "pytorch"
    print(f"[INFO] Using {model_type} Transformer implementation")

    # 检查模型文件是否存在
    if not os.path.exists(FINAL_MODEL_PATH):
        print(f"[ERROR] Model file not found: {FINAL_MODEL_PATH}")
        print(f"[INFO] Please train the model first using transformer_translation.py")
        print(f"[INFO] Current configuration: USE_CUSTOM_TRANSFORMER = {USE_CUSTOM_TRANSFORMER}")
        return

    # 加载验证集
    print(f"[INFO] Loading test data from WMT19 ...")
    src_texts, tgt_texts = load_data(split="validation", sample_size=100)
    print(f"[INFO] Loaded {len(src_texts)} samples.")

    # 初始化分词器
    print("[INFO] Initializing tokenizers ...")
    src_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    tgt_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 构建模型并载入参数
    print(f"[INFO] Building model and loading weights from {FINAL_MODEL_PATH} ...")
    model = TransformerTranslator(
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=device))
        print(f"[INFO] Successfully loaded model from {FINAL_MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
        
    model.eval()
    print("[INFO] Model is ready. Starting translation ...")

    # 推理并收集预测
    hyps = []
    total = len(src_texts)
    for idx, src in enumerate(src_texts, 1):
        print(f"[PROGRESS] Translating {idx}/{total}", end="\r", flush=True)
        hyp = translate(model, src, src_tokenizer, tgt_tokenizer, max_len=MAX_LEN,beam_size=5,length_penalty=0)
        hyps.append(hyp)
    print("\n[INFO] Translation complete.")

    # 新增：展示若干条示例翻译结果
    print("[INFO] Sample translations:")
    for src, hyp, tgt in zip(src_texts[:10], hyps[:10], tgt_texts[:10]):
        print(f"Src: {src}")
        print(f"Hyp: {hyp}")
        print(f"Ref: {tgt}")
        print("-" * 50)

    # 计算 BLEU 
    print("[INFO] Calculating BLEU score ...")
    bleu = sacrebleu.corpus_bleu(hyps, [tgt_texts])
    print(f"[RESULT] BLEU = {bleu.score:.2f}")

if __name__ == "__main__":
    main()
