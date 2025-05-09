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
    # 设备配置（论文未明确但推荐实现）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化数据集（论文5.2节）
    tokenizer_path = 'bpe_tokenizer.json'
    train_dataset = TranslationDataset(
        src_path='data/train.src',
        tgt_path='data/train.tgt',
        tokenizer_path=tokenizer_path
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    # 初始化模型（论文3.2节参数）
    # 在模型初始化时添加完整参数
    model = Transformer(
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        vocab_size=32000,
        dropout=0.1,  # 新增超参数
        max_len=512
    ).to(device)
    
    # 优化器配置（论文5.3节）
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    # 学习率预热（论文5.3节公式）
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min((step+1)**-0.5, (step+1)*4000**-1.5)
    )
    
    # 训练循环（论文4.2节）
    model.train()
    for epoch in range(100):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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