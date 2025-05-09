import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from model import Transformer  # 确保模型文件存在

class TranslationDataset(Dataset):
    def __init__(self, src_path, tgt_path, tokenizer_path, max_length=100):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_length = max_length
        
        # 读取原始数据
        with open(src_path, 'r', encoding='utf-8') as f:
            self.src_texts = [line.strip() for line in f]
        with open(tgt_path, 'r', encoding='utf-8') as f:
            self.tgt_texts = [line.strip() for line in f]

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        # 编码并添加特殊符号（论文3.4节）
        src = self.tokenizer.encode(self.src_texts[idx]).ids
        tgt = self.tokenizer.encode(self.tgt_texts[idx]).ids
        
        # 填充序列（论文5.2节）
        src = src[:self.max_length-2]
        src = [self.tokenizer.token_to_id('<sos>')] + src + [self.tokenizer.token_to_id('<eos>')]
        src += [self.tokenizer.token_to_id('<pad>')] * (self.max_length - len(src))
        
        tgt = tgt[:self.max_length-2]
        tgt = [self.tokenizer.token_to_id('<sos>')] + tgt + [self.tokenizer.token_to_id('<eos>')]
        tgt += [self.tokenizer.token_to_id('<pad>')] * (self.max_length - len(tgt))
        
        return torch.LongTensor(src), torch.LongTensor(tgt)

def train():
    # 设备配置（支持MPS加速）
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    
    # 初始化数据集（论文5.2节）
    tokenizer_path = 'bpe_tokenizer.json'
    # 在文件顶部添加超参数配置
    HYPERPARAMETERS = {
        # 数据参数
        "BATCH_SIZE": 32,
        "MAX_LENGTH": 100,
        "VOCAB_SIZE": 32000,
        
        # 模型参数
        "D_MODEL": 512,
        "NHEAD": 8,
        "NUM_ENCODER_LAYERS": 6,
        "NUM_DECODER_LAYERS": 6,
        "D_FF": 2048,
        "DROPOUT": 0.1,
        "MAX_LEN": 512,
        
        # 优化参数
        "LR": 0.001,
        "BETAS": (0.9, 0.98),
        "EPS": 1e-9,
        "WARMUP_STEPS": 4000,
        "CLIP_GRAD_NORM": 1.0,
        "EPOCHS": 10
    }
    
    # 修改数据集初始化
    train_dataset = TranslationDataset(
        src_path='data/train.src',
        tgt_path='data/train.tgt',
        tokenizer_path=tokenizer_path,
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
    
    # 修改训练循环
    for epoch in range(HYPERPARAMETERS["EPOCHS"]):
        total_loss = 0
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            
            # 前向传播（论文3.2节）
            output = model(src, tgt[:, :-1])
            loss = nn.CrossEntropyLoss(ignore_index=0)(
                output.reshape(-1, 32000),
                tgt[:, 1:].reshape(-1)
            )
            
            # 反向传播（论文5.3节）
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), HYPERPARAMETERS["CLIP_GRAD_NORM"])
            optimizer.step()
            lr_scheduler.step()
            
            total_loss += loss.item()
            
            # 每100批次打印日志
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/100] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')
        
        # 保存检查点（论文未提及但推荐实现）
        torch.save(model.state_dict(), f'transformer_epoch{epoch+1}.pth')
        print(f'Epoch [{epoch+1}/100] Average Loss: {total_loss/len(train_loader):.4f}')

if __name__ == '__main__':
    train()