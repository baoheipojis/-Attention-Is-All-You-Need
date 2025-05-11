import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from transformers import BertTokenizer, XLMRobertaTokenizer
from model import Transformer  # 确保模型文件存在

# 断点续训配置
CHECKPOINT_CONFIG = {
    "RESUME_FROM": "transformer_epoch12.pth",  # 设置为None表示从头开始训练，否则填入检查点文件路径
    "SAVE_DIR": ".",  # 检查点保存目录
    "SAVE_EVERY": 1,  # 每隔多少个epoch保存一次检查点
}

class TranslationDataset(Dataset):
    def __init__(self, src_path, tgt_path, src_tokenizer_name='bert-base-chinese', 
                 tgt_tokenizer_name='bert-base-uncased', max_length=100):
        # 使用预训练的tokenizer，为中文和英文分别加载
        self.src_tokenizer = BertTokenizer.from_pretrained(src_tokenizer_name)
        self.tgt_tokenizer = BertTokenizer.from_pretrained(tgt_tokenizer_name)
        self.max_length = max_length
        
        # 读取原始数据
        with open(src_path, 'r', encoding='utf-8') as f:
            self.src_texts = [line.strip() for line in f]
        with open(tgt_path, 'r', encoding='utf-8') as f:
            self.tgt_texts = [line.strip() for line in f]

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        # 对源语言(中文)进行编码
        src_encoding = self.src_tokenizer(
            self.src_texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 对目标语言(英文)进行编码
        tgt_encoding = self.tgt_tokenizer(
            self.tgt_texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return src_encoding['input_ids'].squeeze(), tgt_encoding['input_ids'].squeeze()

def save_checkpoint(model, optimizer, lr_scheduler, epoch, hyperparams, filename):
    """保存检查点，包含模型、优化器、调度器状态和训练信息"""
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
        'epoch': epoch,
        'hyperparams': hyperparams
    }
    torch.save(checkpoint, filename)
    print(f"检查点已保存: {filename}")

def load_checkpoint(filename, model, optimizer=None, lr_scheduler=None):
    """加载检查点，恢复模型、优化器、调度器状态和训练信息"""
    if not os.path.exists(filename):
        print(f"检查点文件不存在: {filename}")
        return None, 0, {}
    
    checkpoint = torch.load(filename)
    
    # 检测旧格式的检查点（直接保存state_dict）
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 新格式
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        hyperparams = checkpoint.get('hyperparams', {})
    else:
        # 旧格式，直接是模型的state_dict
        model.load_state_dict(checkpoint)
        epoch = 0  # 因为旧格式不保存epoch信息，所以从0开始
        hyperparams = {}
        print("检测到旧格式的检查点文件，仅恢复模型参数")
    
    print(f"已从检查点恢复: {filename}")
    return model, epoch, hyperparams

def train():
    # 设备配置（支持MPS加速）
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"使用设备: {device}")
    
    # 预训练tokenizer配置
    src_tokenizer_name = 'bert-base-chinese'  # 中文tokenizer
    tgt_tokenizer_name = 'bert-base-uncased'  # 英文tokenizer
    
    # 获取预训练tokenizer的词汇量大小
    src_tokenizer = BertTokenizer.from_pretrained(src_tokenizer_name)
    tgt_tokenizer = BertTokenizer.from_pretrained(tgt_tokenizer_name)
    src_vocab_size = len(src_tokenizer)
    tgt_vocab_size = len(tgt_tokenizer)
    vocab_size = max(src_vocab_size, tgt_vocab_size)  # 选择较大的词汇量
    
    # 更新超参数配置
    HYPERPARAMETERS = {
        # 数据参数
        "BATCH_SIZE": 64,
        "MAX_LENGTH": 100,
        "VOCAB_SIZE": vocab_size,
        
        # 模型参数
        "D_MODEL": 128,
        "NHEAD": 4,
        "NUM_ENCODER_LAYERS": 6,
        "NUM_DECODER_LAYERS": 6,
        "D_FF": 512,
        "DROPOUT": 0.1,
        "MAX_LEN": 512,
        
        # 优化参数
        "LR": 0.001,
        "BETAS": (0.9, 0.98),
        "EPS": 1e-9,
        "WARMUP_STEPS": 4000,
        "CLIP_GRAD_NORM": 1.0,
        "EPOCHS": 100,
        
        # Tokenizer参数
        "PAD_TOKEN_ID": src_tokenizer.pad_token_id
    }
    
    # 修改数据集初始化
    train_dataset = TranslationDataset(
        src_path='data/train.src',  # 修改为中文数据文件路径
        tgt_path='data/train.tgt',  # 修改为英文数据文件路径
        src_tokenizer_name=src_tokenizer_name,
        tgt_tokenizer_name=tgt_tokenizer_name,
        max_length=HYPERPARAMETERS["MAX_LENGTH"]
    )
    train_loader = DataLoader(train_dataset, batch_size=HYPERPARAMETERS["BATCH_SIZE"], shuffle=True)
    
    # 修改模型初始化
    model = Transformer(
        d_model=HYPERPARAMETERS["D_MODEL"],
        nhead=HYPERPARAMETERS["NHEAD"],
        num_encoder_layers=HYPERPARAMETERS["NUM_ENCODER_LAYERS"],
        num_decoder_layers=HYPERPARAMETERS["NUM_DECODER_LAYERS"],
        d_ff=HYPERPARAMETERS["D_FF"],
        vocab_size=HYPERPARAMETERS["VOCAB_SIZE"],
        dropout=HYPERPARAMETERS["DROPOUT"],
        max_len=HYPERPARAMETERS["MAX_LEN"]
    ).to(device)
    
    # 修改优化器配置
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=HYPERPARAMETERS["LR"],
        betas=HYPERPARAMETERS["BETAS"],
        eps=HYPERPARAMETERS["EPS"]
    )
    
    # 修改学习率预热
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min((step+1)**-0.5, (step+1)*HYPERPARAMETERS["WARMUP_STEPS"]**-1.5)
    )
    
    # 从检查点恢复训练（如果指定）
    start_epoch = 0
    if CHECKPOINT_CONFIG["RESUME_FROM"]:
        model, start_epoch, loaded_params = load_checkpoint(
            CHECKPOINT_CONFIG["RESUME_FROM"], 
            model, 
            optimizer, 
            lr_scheduler
        )
        # 恢复后需要从下一个epoch开始
        start_epoch += 1
        print(f"从epoch {start_epoch}开始继续训练")
    
    # 确保检查点保存目录存在
    os.makedirs(CHECKPOINT_CONFIG["SAVE_DIR"], exist_ok=True)
    
    # 修改训练循环
    for epoch in range(start_epoch, HYPERPARAMETERS["EPOCHS"]):
        model.train()
        total_loss = 0
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            
            output = model(src, tgt[:, :-1])
            
            loss = nn.CrossEntropyLoss(ignore_index=HYPERPARAMETERS["PAD_TOKEN_ID"])(
                output.reshape(-1, HYPERPARAMETERS["VOCAB_SIZE"]),
                tgt[:, 1:].reshape(-1)
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), HYPERPARAMETERS["CLIP_GRAD_NORM"])
            optimizer.step()
            lr_scheduler.step()
            
            total_loss += loss.item()
            
            # 每100批次打印日志
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{HYPERPARAMETERS["EPOCHS"]}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')
        
        # 保存检查点
        if (epoch + 1) % CHECKPOINT_CONFIG["SAVE_EVERY"] == 0:
            checkpoint_path = os.path.join(CHECKPOINT_CONFIG["SAVE_DIR"], f'transformer_epoch{epoch+1}.pth')
            save_checkpoint(model, optimizer, lr_scheduler, epoch, HYPERPARAMETERS, checkpoint_path)
        
        print(f'Epoch [{epoch+1}/{HYPERPARAMETERS["EPOCHS"]}] Average Loss: {total_loss/len(train_loader):.4f}')

if __name__ == '__main__':
    train()