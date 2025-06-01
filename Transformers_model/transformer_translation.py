import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import time, os, math

# 导入自定义的Transformer模型
from model import Transformer
from data import load_data  # Importing the new data loader

#--------------------------------
# 模型选择配置
#--------------------------------
USE_CUSTOM_TRANSFORMER = False  # True: 使用自定义Transformer, False: 使用PyTorch nn.Transformer

#--------------------------------
# 超参数配置
#--------------------------------
# 数据处理参数
MAX_LEN = 128          # 序列最大长度
BATCH_SIZE = 64         # 批处理大小

# 模型架构参数
D_MODEL = 128         # 模型维度
N_HEAD = 8             # 注意力头数
NUM_ENCODER_LAYERS = 6 # 编码器层数
NUM_DECODER_LAYERS = 6 # 解码器层数
DIM_FEEDFORWARD = 512  # 前馈网络维度
DROPOUT = 0.1          # Dropout概率

# 训练参数
LEARNING_RATE = 0.001  # 学习率
BETAS = (0.9, 0.98)    # Adam优化器的beta参数
EPSILON = 1e-9         # Adam优化器的epsilon参数
NUM_EPOCHS = 500       # 训练轮次
RESUME_EPOCH = 122     # 手动设置续训起始轮次，如果为0则从头开始
GRAD_CLIP = 1.0        # 梯度裁剪阈值
EARLY_STOP_THRESHOLD = 0.1  # 早停阈值
EVAL_EVERY = 1       # 每多少轮进行一次评估
WARMUP_STEPS = 4000    # 学习率预热步数
WARMUP_FACTOR = 0.1    # 预热开始因子，初始学习率 = LEARNING_RATE * WARMUP_FACTOR
LABEL_SMOOTHING = 0.1  # 标签平滑参数

# 文件路径
SRC_FILE = "data/val.src"
TGT_FILE = "data/val.tgt"
MODEL_DIR = os.path.dirname(SRC_FILE)
model_suffix = "" if USE_CUSTOM_TRANSFORMER else "_pytorch"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, f"best_transformer_model{model_suffix}.pt")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, f"transformer_translation_model{model_suffix}.pt")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, f"training_checkpoint{model_suffix}.pt")

# add a directory to hold per-epoch logs
LOG_DIR = os.path.join(MODEL_DIR, "logs")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# 使用自定义Transformer模型替换原TransformerTranslator类
class TransformerTranslator(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=D_MODEL, nhead=N_HEAD, 
                 num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS, 
                 dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT):
        super(TransformerTranslator, self).__init__()
        
        self.use_custom = USE_CUSTOM_TRANSFORMER
        self.tgt_vocab_size = tgt_vocab_size
        
        if USE_CUSTOM_TRANSFORMER:
            print("[INFO] Using custom Transformer implementation")
            # 使用自定义的Transformer模型
            common_vocab_size = max(src_vocab_size, tgt_vocab_size)
            self.transformer = Transformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                d_ff=dim_feedforward,
                vocab_size=common_vocab_size,
                max_len=MAX_LEN,
                dropout=dropout
            )
            self.common_vocab_size = common_vocab_size
        else:
            print("[INFO] Using PyTorch nn.Transformer implementation")
            # 使用PyTorch内置的Transformer
            self.src_embedding = nn.Embedding(src_vocab_size, d_model)
            self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
            self.pos_encoding = nn.Parameter(torch.randn(MAX_LEN, d_model))
            
            self.transformer = nn.Transformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            
            self.output_projection = nn.Linear(d_model, tgt_vocab_size)
            self.common_vocab_size = tgt_vocab_size
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        if self.use_custom:
            # 自定义Transformer模型不需要外部传入掩码，它会在内部创建
            output = self.transformer(src, tgt)
            
            # 如果模型的输出维度大于目标词汇表维度，需要截断
            if output.shape[-1] > self.tgt_vocab_size:
                output = output[..., :self.tgt_vocab_size]
                
            return output
        else:
            # PyTorch nn.Transformer需要手动处理embedding和掩码
            batch_size, src_len = src.shape
            tgt_len = tgt.shape[1]
            
            # 添加位置编码
            src_emb = self.src_embedding(src) + self.pos_encoding[:src_len].unsqueeze(0)
            tgt_emb = self.tgt_embedding(tgt) + self.pos_encoding[:tgt_len].unsqueeze(0)
            
            # 创建掩码
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(src.device)
            
            # Transformer前向传播
            output = self.transformer(
                src_emb, tgt_emb, 
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask
            )
            
            # 输出投影
            output = self.output_projection(output)
            
            return output

# 简化的掩码生成函数，仅保留给现有代码使用，实际上新的Transformer会自己处理掩码
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.to(device)

def create_masks(src, tgt, src_pad_idx, tgt_pad_idx):
    # 这个函数仍然保留，但在forward中我们不会使用这些掩码
    src_padding_mask = (src == src_pad_idx)
    tgt_padding_mask = (tgt == tgt_pad_idx)
    tgt_seq_len = tgt.size(1)
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = None
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# Training loop without custom warmup calls
def train_epoch(model, dataloader, optimizer, criterion, src_pad_idx, tgt_pad_idx):
    model.train()
    total_loss = 0
    for idx, batch in enumerate(dataloader):
        src = batch['src_input_ids'].to(device)
        tgt = batch['tgt_input_ids'].to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        src = torch.clamp(src, 0, model.common_vocab_size-1)
        tgt_input = torch.clamp(tgt_input, 0, model.common_vocab_size-1)
        output = model(src, tgt_input)
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)
        loss = criterion(output, tgt_output)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        # print batch progress with carriage return
        print(f"Batch {idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}\r", end="", flush=True)

    return total_loss / len(dataloader)

# 修改translate函数以适应新模型
def translate(model, src_text, src_tokenizer, tgt_tokenizer, max_len=MAX_LEN, beam_size=1, length_penalty=0.6):
    """使用束搜索进行翻译
    
    Args:
        model: 翻译模型
        src_text: 源文本
        src_tokenizer: 源语言分词器
        tgt_tokenizer: 目标语言分词器
        max_len: 最大生成长度
        beam_size: 束宽，默认为1（相当于贪心搜索）
        length_penalty: 长度惩罚系数，0为不惩罚，<1时倾向于短句子，>1时倾向于长句子
    
    Returns:
        翻译后的文本
    """
    model.eval()
    
    # 编码输入文本
    tokens = src_tokenizer(
        src_text,
        return_tensors='pt',
        max_length=max_len,
        padding='max_length',
        truncation=True
    )
    src = tokens['input_ids'].to(device)
    src = torch.clamp(src, 0, model.common_vocab_size-1)
    
    # 特殊标记ID
    sos_id = tgt_tokenizer.cls_token_id
    eos_id = tgt_tokenizer.sep_token_id
    
    with torch.no_grad():
        # 如果束宽为1，使用贪心搜索（更高效）
        if beam_size == 1:
            # 从<sos>标记开始
            tgt = torch.ones(1, 1).fill_(sos_id).type_as(src).to(device)
            tgt = torch.clamp(tgt, 0, model.common_vocab_size-1)
            
            for i in range(max_len - 1):
                output = model(src, tgt)
                prob = output[:, -1, :]  # 获取最后一个位置的预测
                _, next_word = torch.max(prob, dim=1)
                next_token = next_word.unsqueeze(1)
                
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # 如果生成了结束标记，停止生成
                if next_token.item() == eos_id:
                    break
            
            return tgt_tokenizer.decode(tgt[0].tolist(), skip_special_tokens=True)
        
        # 束搜索实现
        # 初始化候选序列 - [(序列, 序列得分), ...]
        k_prev_words = torch.ones(beam_size, 1).long().fill_(sos_id).to(device)
        k_prev_words = torch.clamp(k_prev_words, 0, model.common_vocab_size-1)
        
        # 候选序列的得分
        k_scores = torch.zeros(beam_size, device=device)
        
        # 完成的候选序列
        complete_seqs = []
        complete_seqs_scores = []
        
        # 逐步生成序列
        step = 1
        while True:
            # 扩展每个候选序列
            curr_size = k_prev_words.size(0)  # 当前候选数量
            src_expanded = src.expand(curr_size, -1)  # (k, src_len)
            
            # 对所有候选进行前向计算
            output = model(src_expanded, k_prev_words)  # (k, step, vocab_size)
            logits = output[:, -1, :]  # 获取最后一步的输出 (k, vocab_size)
            
            # 对数概率和之前的分数相加
            log_probs = torch.log_softmax(logits, dim=-1)
            log_probs = log_probs + k_scores.unsqueeze(1)  # (k, vocab_size)
            
            # 对于第一步，所有候选的得分都是0，只考虑从第一个序列扩展的结果
            if step == 1:
                log_probs = log_probs[0].unsqueeze(0)  # (1, vocab_size)
            
            # 展平以便获取前k个候选
            vocab_size = log_probs.size(-1)
            log_probs = log_probs.view(-1)  # (k * vocab_size)
            
            # 获取分数最高的k个候选词
            k_scores, k_indices = torch.topk(log_probs, k=min(beam_size, log_probs.size(0)))
            
            # 计算这些词属于哪个前序列
            prev_seq_indices = k_indices // vocab_size  # 商给出之前的候选序列索引
            next_word_indices = k_indices % vocab_size  # 余数给出词索引
            
            # 构建新的候选序列
            k_next_words = []
            for i, (prev_seq_idx, next_word_idx) in enumerate(zip(prev_seq_indices, next_word_indices)):
                # 添加到新候选集
                new_seq = torch.cat([k_prev_words[prev_seq_idx], next_word_idx.unsqueeze(0)], dim=0)
                k_next_words.append(new_seq)
                
                # 如果生成了结束标记，将其添加到完成序列中
                if next_word_idx.item() == eos_id:
                    complete_seqs.append(new_seq)
                    
                    # 应用长度惩罚: (5+len(seq))^length_penalty / (5+1)^length_penalty
                    # 常数5是为了减少对非常短序列的过度惩罚
                    lp = ((5 + len(new_seq)) ** length_penalty) / ((5 + 1) ** length_penalty)
                    complete_seqs_scores.append(k_scores[i].item() / lp)
            
            # 如果所有候选序列都已完成
            if len(complete_seqs) >= beam_size:
                break
            
            # 更新候选序列
            k_prev_words = torch.stack(k_next_words)
            
            # 增加步数
            step += 1
            
            # 如果达到最大长度，结束搜索
            if step > max_len:
                break
        
        # 选择得分最高的完成序列，已应用长度惩罚
        if complete_seqs:
            # 从完成的序列中选择最好的
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
        else:
            # 如果没有完成的序列，选择当前得分最高的候选
            # 应用相同的长度惩罚策略
            penalized_scores = []
            for i, words in enumerate(k_prev_words):
                lp = ((5 + len(words)) ** length_penalty) / ((5 + 1) ** length_penalty)
                penalized_scores.append(k_scores[i].item() / lp)
            
            best_idx = penalized_scores.index(max(penalized_scores))
            seq = k_prev_words[best_idx]
    
    # 解码生成的序列
    return tgt_tokenizer.decode(seq.tolist(), skip_special_tokens=True)

def save_checkpoint(epoch, model, optimizer, scheduler, lr_plateau, best_loss, checkpoint_path):
    """保存训练检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'lr_plateau_state_dict': lr_plateau.state_dict(),
        'best_loss': best_loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, lr_plateau):
    """加载训练检查点"""
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        lr_plateau.load_state_dict(checkpoint['lr_plateau_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        
        print(f"Resumed from epoch {checkpoint['epoch']}, best loss: {best_loss:.4f}")
        return start_epoch, best_loss
    elif RESUME_EPOCH > 0 and os.path.exists(BEST_MODEL_PATH):
        # 如果没有checkpoint但有best model且设置了续训轮次
        print(f"Loading best model from {BEST_MODEL_PATH}")
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        print(f"Resuming from epoch {RESUME_EPOCH} (optimizer and scheduler states reset)")
        print("Warning: Learning rate schedule will restart from beginning")
        return RESUME_EPOCH, 4.2428  # 使用你提到的最佳loss值
    else:
        print("No checkpoint found, starting from scratch")
        return 0, float('inf')

def main():
    # 加载训练和验证数据
    src_texts, tgt_texts = load_data(split="train", sample_size=200000)
    print(f"Loaded {len(src_texts)} translation pairs")
    
    val_src_texts, val_tgt_texts = load_data(split="validation", sample_size=500)
    print(f"Loaded {len(val_src_texts)} validation pairs")
    
    src_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    tgt_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    dataset = TranslationDataset(src_texts, tgt_texts, src_tokenizer, tgt_tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    src_vocab_size = src_tokenizer.vocab_size
    tgt_vocab_size = tgt_tokenizer.vocab_size
    
    model_type = "custom" if USE_CUSTOM_TRANSFORMER else "pytorch"
    print(f"Creating {model_type} Transformer model...")
    
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
    
    print(f"{model_type.capitalize()} model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # use built-in label smoothing
    criterion = nn.CrossEntropyLoss(
        ignore_index=tgt_tokenizer.pad_token_id,
        label_smoothing=LABEL_SMOOTHING
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=EPSILON
    )

    # total training steps for scheduler
    total_steps = len(dataloader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )

    # optional ReduceLROnPlateau if desired
    lr_plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 加载检查点（如果存在）
    start_epoch, best_loss = load_checkpoint(CHECKPOINT_PATH, model, optimizer, scheduler, lr_plateau)

    for epoch in range(start_epoch, NUM_EPOCHS):
        start = time.time()
        train_loss = train_epoch(
            model, dataloader, optimizer, criterion,
            src_tokenizer.pad_token_id, tgt_tokenizer.pad_token_id
        )

        # step schedulers
        scheduler.step()
        lr_plateau.step(train_loss)

        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {train_loss:.4f} | Time: {elapsed:.2f}s")

        # write this epoch's loss to a new log file without overwriting
        log_filename = os.path.join(
            LOG_DIR,
            f"epoch_{epoch+1}_{int(time.time())}.log"
        )
        with open(log_filename, "w", encoding="utf-8") as log_f:
            log_f.write(f"Epoch: {epoch+1}, Loss: {train_loss:.4f}\n")
        
        if train_loss < best_loss:
            best_loss = train_loss
            # 统一保存格式：仅保存模型参数
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"New best model saved with loss: {best_loss:.4f}")
        
        # 每5个epoch保存一次检查点，避免频繁IO
        if (epoch + 1) % 5 == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, lr_plateau, best_loss, CHECKPOINT_PATH)
        
        if (epoch + 1) % EVAL_EVERY == 0:
            print("\nTesting translations:")
            for src_text in src_texts[:3]:
                translation = translate(model, src_text, src_tokenizer, tgt_tokenizer)
                print(f"Source: {src_text}")
                print(f"Translation: {translation}")
                print("-" * 40)
            
            if train_loss < EARLY_STOP_THRESHOLD:
                print(f"Loss {train_loss:.4f} is below threshold. Stopping training.")
                break
    
    # 训练结束前保存最终检查点
    save_checkpoint(epoch, model, optimizer, scheduler, lr_plateau, best_loss, CHECKPOINT_PATH)
    
    # 训练结束后，同样仅保存模型参数
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"Final model saved to {FINAL_MODEL_PATH}")
    
    print("\nFinal translations:")
    for i, src_text in enumerate(src_texts):
        translation = translate(model, src_text, src_tokenizer, tgt_tokenizer)
        print(f"Source: {src_text}")
        print(f"Expected: {tgt_texts[i]}")
        print(f"Translation: {translation}")
        print("-" * 40)

if __name__ == "__main__":
    main()
