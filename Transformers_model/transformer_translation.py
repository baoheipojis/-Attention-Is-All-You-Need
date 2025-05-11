import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import math
import time
import os

#--------------------------------
# 超参数配置
#--------------------------------
# 数据处理参数
MAX_LEN = 128          # 序列最大长度
BATCH_SIZE = 3         # 批处理大小

# 模型架构参数
D_MODEL = 256          # 模型维度
N_HEAD = 8             # 注意力头数
NUM_ENCODER_LAYERS = 3 # 编码器层数
NUM_DECODER_LAYERS = 3 # 解码器层数
DIM_FEEDFORWARD = 512  # 前馈网络维度
DROPOUT = 0.1          # Dropout概率

# 训练参数
LEARNING_RATE = 0.001  # 学习率
BETAS = (0.9, 0.98)    # Adam优化器的beta参数
EPSILON = 1e-9         # Adam优化器的epsilon参数
NUM_EPOCHS = 500       # 训练轮次
GRAD_CLIP = 1.0        # 梯度裁剪阈值
EARLY_STOP_THRESHOLD = 0.1  # 早停阈值
EVAL_EVERY = 50        # 每多少轮进行一次评估

# 文件路径
SRC_FILE = r"c:\Users\who65\OneDrive - 南京大学\大三上课程\Training and Practice of Scientific Research\科研\Transformer_Reproduction\Transformers_model\data\little.src"
TGT_FILE = r"c:\Users\who65\OneDrive - 南京大学\大三上课程\Training and Practice of Scientific Research\科研\Transformer_Reproduction\Transformers_model\data\little.tgt"
MODEL_DIR = os.path.dirname(SRC_FILE)
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_transformer_model.pt")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "transformer_translation_model.pt")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
def load_data(src_file, tgt_file):
    with open(src_file, 'r', encoding='utf-8') as f:
        src_data = [line.strip() for line in f.readlines()]
    
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_data = [line.strip() for line in f.readlines()]
    
    return src_data, tgt_data

# Dataset class for translation
class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, max_len=MAX_LEN):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        
        src_encoded = self.src_tokenizer(
            src_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        tgt_encoded = self.tgt_tokenizer(
            tgt_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'src_input_ids': src_encoded['input_ids'].squeeze(),
            'src_attention_mask': src_encoded['attention_mask'].squeeze(),
            'tgt_input_ids': tgt_encoded['input_ids'].squeeze(),
            'tgt_attention_mask': tgt_encoded['attention_mask'].squeeze()
        }

# Fixed Positional encoding for batch_first=True
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=DROPOUT, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Transformer model for translation
class TransformerTranslator(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=D_MODEL, nhead=N_HEAD, 
                 num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS, 
                 dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT):
        super(TransformerTranslator, self).__init__()
        
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_decoder_layers
        )
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.0)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        src_emb = self.src_embed(src) * math.sqrt(self.d_model)
        src_emb = self.positional_encoding(src_emb)
        
        tgt_emb = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.positional_encoding(tgt_emb)
        
        memory = self.transformer_encoder(
            src_emb, 
            mask=src_mask, 
            src_key_padding_mask=src_padding_mask
        )
        
        output = self.transformer_decoder(
            tgt_emb, 
            memory,
            tgt_mask=tgt_mask, 
            memory_mask=None,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        
        return self.fc_out(output)

# Generate masks for training
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.to(device)

def create_masks(src, tgt, src_pad_idx, tgt_pad_idx):
    src_padding_mask = (src == src_pad_idx)
    tgt_padding_mask = (tgt == tgt_pad_idx)
    tgt_seq_len = tgt.size(1)
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = None
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# Training function
def train_epoch(model, dataloader, optimizer, criterion, src_pad_idx, tgt_pad_idx):
    model.train()
    losses = 0
    
    for batch in dataloader:
        src = batch['src_input_ids'].to(device)
        tgt = batch['tgt_input_ids'].to(device)
        
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_masks(
            src, tgt_input, src_pad_idx, tgt_pad_idx)
        
        output = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
        
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)
        
        loss = criterion(output, tgt_output)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        
        losses += loss.item()
    
    return losses / len(dataloader)

# Improved translate function for inference
def translate(model, src_text, src_tokenizer, tgt_tokenizer, max_len=MAX_LEN):
    model.eval()
    
    tokens = src_tokenizer(
        src_text,
        return_tensors='pt',
        max_length=max_len,
        padding='max_length',
        truncation=True
    )
    src = tokens['input_ids'].to(device)
    src_padding_mask = (src == src_tokenizer.pad_token_id)
    
    tgt = torch.ones(1, 1).fill_(tgt_tokenizer.cls_token_id).type_as(src).to(device)
    
    for i in range(max_len - 1):
        tgt_seq_len = tgt.size(1)
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
        tgt_padding_mask = (tgt == tgt_tokenizer.pad_token_id)
        
        with torch.no_grad():
            output = model(src, tgt, None, tgt_mask, src_padding_mask, tgt_padding_mask)
            prob = output[:, -1, :]
            _, next_word = torch.max(prob, dim=1)
            next_token = next_word.unsqueeze(1)
        
        tgt = torch.cat([tgt, next_token], dim=1)
        
        if next_token.item() == tgt_tokenizer.sep_token_id:
            break
    
    output_text = tgt_tokenizer.decode(tgt[0].tolist(), skip_special_tokens=True)
    
    return output_text

def main():
    src_file = SRC_FILE
    tgt_file = TGT_FILE
    
    src_texts, tgt_texts = load_data(src_file, tgt_file)
    print(f"Loaded {len(src_texts)} translation pairs")
    
    src_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    tgt_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    dataset = TranslationDataset(src_texts, tgt_texts, src_tokenizer, tgt_tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    src_vocab_size = src_tokenizer.vocab_size
    tgt_vocab_size = tgt_tokenizer.vocab_size
    
    model = TransformerTranslator(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=BETAS, eps=EPSILON)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    num_epochs = NUM_EPOCHS
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss = train_epoch(
            model, dataloader, optimizer, criterion,
            src_tokenizer.pad_token_id, tgt_tokenizer.pad_token_id
        )
        
        scheduler.step(train_loss)
        
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f} | Time: {elapsed:.2f}s")
        
        if train_loss < best_loss:
            best_loss = train_loss
            model_path = BEST_MODEL_PATH
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, model_path)
            print(f"New best model saved with loss: {best_loss:.4f}")
        
        if (epoch + 1) % EVAL_EVERY == 0:
            print("\nTesting translations:")
            for src_text in src_texts:
                translation = translate(model, src_text, src_tokenizer, tgt_tokenizer)
                print(f"Source: {src_text}")
                print(f"Translation: {translation}")
                print("-" * 40)
            
            if train_loss < EARLY_STOP_THRESHOLD:
                print(f"Loss {train_loss:.4f} is below threshold. Stopping training.")
                break
    
    model_path = FINAL_MODEL_PATH
    torch.save(model.state_dict(), model_path)
    print(f"Final model saved to {model_path}")
    
    print("\nFinal translations:")
    for i, src_text in enumerate(src_texts):
        translation = translate(model, src_text, src_tokenizer, tgt_tokenizer)
        print(f"Source: {src_text}")
        print(f"Expected: {tgt_texts[i]}")
        print(f"Translation: {translation}")
        print("-" * 40)

if __name__ == "__main__":
    main()
