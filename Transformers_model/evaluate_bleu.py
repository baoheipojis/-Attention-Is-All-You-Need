import torch
from transformers import BertTokenizer
import sacrebleu

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
)
FINAL_MODEL_PATH = BEST_MODEL_PATH
SRC_FILE = "data/val.src"
TGT_FILE = "data/val.tgt"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 加载验证集
    print(f"[INFO] Loading data from {SRC_FILE} and {TGT_FILE} ...")
    src_texts, tgt_texts = load_data(SRC_FILE, TGT_FILE)
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
    model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=device))
    model.eval()
    print("[INFO] Model is ready. Starting translation ...")

    # 推理并收集预测
    hyps = []
    total = len(src_texts)
    for idx, src in enumerate(src_texts, 1):
        print(f"[PROGRESS] Translating {idx}/{total}", end="\r", flush=True)
        hyp = translate(model, src, src_tokenizer, tgt_tokenizer, max_len=MAX_LEN,beam_size=10,length_penalty=0.6)
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
