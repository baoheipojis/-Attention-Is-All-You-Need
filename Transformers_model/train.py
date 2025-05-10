import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from transformers import BertTokenizer, XLMRobertaTokenizer
from model import Transformer  # 确保模型文件存在

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

def train():
    # 设备配置（支持MPS加速）
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    
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
        "EPOCHS": 10,
        
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
    
    # 修改训练循环
    for epoch in range(HYPERPARAMETERS["EPOCHS"]):
        total_loss = 0
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            
            # 这里的 tgt[:, :-1] 表示取目标序列的所有样本（第一维），但每个序列只取除了最后一个 token 以外的所有 token（第二维）
            # 这是因为在训练时，解码器的输入应该是目标序列"移位一位"的结果:
            # - 解码器输入: <sos> token1 token2 ... tokenN-1 （不包含最后的<eos>）
            # - 期望输出: token1 token2 ... tokenN-1 <eos> （不包含开头的<sos>）
            # 也就是用前面的token去预测下一个token
            output = model(src, tgt[:, :-1])
            
            # 然后，损失函数会比较模型输出和 tgt[:, 1:] (不包括开头的<sos>)
            # 实现了经典的"teacher forcing"训练方式
            loss = nn.CrossEntropyLoss(ignore_index=HYPERPARAMETERS["PAD_TOKEN_ID"])(
                output.reshape(-1, HYPERPARAMETERS["VOCAB_SIZE"]),
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
                print(f'Epoch [{epoch+1}/10] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')
        
        # 保存检查点（论文未提及但推荐实现）
        torch.save(model.state_dict(), f'transformer_epoch{epoch+1}.pth')
        print(f'Epoch [{epoch+1}/10] Average Loss: {total_loss/len(train_loader):.4f}')

if __name__ == '__main__':
    train()