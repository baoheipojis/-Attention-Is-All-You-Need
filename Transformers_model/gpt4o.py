import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer

import math

# ==== Config ====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 10
MAX_LEN = 128
MODEL_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 6
FF_DIM = 2048
DROPOUT = 0.1

# ==== Load Dataset ====
dataset = load_dataset("iwslt2017", "iwslt2017-en-de")

# ==== Tokenizer ====
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
PAD_IDX = tokenizer.pad_token_id

def tokenize_function(batch):
    model_inputs = tokenizer(batch["translation"]["en"], truncation=True, padding="max_length", max_length=MAX_LEN)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["translation"]["de"], truncation=True, padding="max_length", max_length=MAX_LEN)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=BATCH_SIZE, shuffle=True)

# ==== Transformer ====

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, MODEL_DIM)
        self.pos_encoder = PositionalEncoding(MODEL_DIM, DROPOUT)
        self.transformer = nn.Transformer(
            d_model=MODEL_DIM,
            nhead=NUM_HEADS,
            num_encoder_layers=NUM_LAYERS,
            num_decoder_layers=NUM_LAYERS,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT
        )
        self.fc_out = nn.Linear(MODEL_DIM, vocab_size)

    def forward(self, src, tgt):
        src_mask = self.generate_square_subsequent_mask(src.size(1)).to(DEVICE)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(DEVICE)

        src = self.embedding(src) * math.sqrt(MODEL_DIM)
        tgt = self.embedding(tgt) * math.sqrt(MODEL_DIM)

        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        output = self.transformer(src.transpose(0, 1), tgt.transpose(0, 1),
                                  src_mask=src_mask, tgt_mask=tgt_mask)
        return self.fc_out(output.transpose(0, 1))

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

# ==== Label Smoothing Loss ====

def label_smoothing_loss(pred, target, smoothing=0.1):
    pred = pred.view(-1, pred.size(-1))
    target = target.view(-1)
    confidence = 1.0 - smoothing
    log_probs = F.log_softmax(pred, dim=-1)
    nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
    smooth_loss = -log_probs.mean(dim=-1)
    loss = confidence * nll_loss + smoothing * smooth_loss
    mask = target != PAD_IDX
    loss = loss * mask
    return loss.sum() / mask.sum()

# ==== Training ====

model = TransformerModel(len(tokenizer)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        src = batch["input_ids"].to(DEVICE)
        tgt = batch["labels"].to(DEVICE)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input)
        loss = label_smoothing_loss(logits, tgt_output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_dataloader):.4f}")
