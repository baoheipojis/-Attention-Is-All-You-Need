import sacrebleu

def compute_bleu_sacrebleu(reference, hypothesis):
    # 将输入字符串转换为 sacrebleu 所需的格式
    references = [[reference.strip()]]
    hypotheses = [hypothesis.strip()]

    score = sacrebleu.corpus_bleu(hypotheses, references)
    return score.score

if __name__ == "__main__":
    ref = "The marshmallow has to be on top"
    hyp = "they have to put it on the sugar"
    bleu_score = compute_bleu_sacrebleu(ref, hyp)
    print(f"BLEU score: {bleu_score:.4f}")