import torch
import torch.nn as nn
import os
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from transformers import BertTokenizer, XLMRobertaTokenizer
from model import Transformer  # 确保模型文件存在

# 确保nltk数据可用
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 断点续训配置
CHECKPOINT_CONFIG = {
    "RESUME_FROM": None,  # 设置为None表示从头开始训练，否则填入检查点文件路径
    "SAVE_DIR": ".",  # 检查点保存目录
    "SAVE_EVERY": 100,  # 每隔多少个epoch保存一次检查点
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

def load_checkpoint(filename, model=None, optimizer=None, lr_scheduler=None, device=None):
    """加载检查点，支持新旧两种格式"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(filename):
        print(f"检查点文件不存在: {filename}")
        return None, 0, {}
    
    checkpoint = torch.load(filename, map_location=device)
    
    # 检测是否为新格式的检查点（包含model_state_dict等键）
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("检测到新格式的检查点文件")
        model_state_dict = checkpoint['model_state_dict']
        
        if model is not None:
            model.load_state_dict(model_state_dict)
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        hyperparams = checkpoint.get('hyperparams', {})
        print(f"加载的检查点来自epoch: {epoch}")
    else:
        # 旧格式直接是模型权重
        print("检测到旧格式的检查点文件")
        model_state_dict = checkpoint
        epoch = 0
        hyperparams = {}
        
        if model is not None:
            model.load_state_dict(model_state_dict)
    
    print(f"已从检查点恢复: {filename}")
    
    if model is not None:
        return model, epoch, hyperparams
    else:
        return model_state_dict, hyperparams

def calculate_bleu(model, data_loader, src_tokenizer, tgt_tokenizer, device, max_samples=None):
    """计算数据集的BLEU分数"""
    model.eval()
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for i, (src, tgt) in enumerate(tqdm(data_loader, desc="Calculating BLEU")):
            if max_samples and i >= max_samples:
                break
                
            src = src.to(device)
            
            # 逐条生成翻译结果
            with torch.no_grad():
                # 生成序列的起始标记
                sos_id = tgt_tokenizer.cls_token_id
                eos_id = tgt_tokenizer.sep_token_id
                
                # 初始化目标序列为起始标记
                batch_size = src.size(0)
                tgt_seq = torch.ones(batch_size, 1).fill_(sos_id).long().to(device)
                
                max_length = 100  # 最大生成长度
                
                # 自回归生成翻译
                for _ in range(max_length):
                    output = model(src, tgt_seq)
                    next_word = output[:, -1, :].argmax(dim=-1, keepdim=True)
                    tgt_seq = torch.cat([tgt_seq, next_word], dim=1)
                    
                    # 检查是否所有序列都已生成结束标记
                    if ((next_word == eos_id).sum() == batch_size) or (tgt_seq.size(1) >= max_length):
                        break
            
            # 处理生成的序列
            for j in range(batch_size):
                # 获取原始目标序列（去除特殊标记）
                ref_ids = tgt[j].tolist()
                # 移除padding和特殊标记
                ref_ids = [id for id in ref_ids if id != tgt_tokenizer.pad_token_id and 
                          id != sos_id and id != eos_id]
                ref_tokens = tgt_tokenizer.convert_ids_to_tokens(ref_ids)
                
                # 获取生成的序列
                hyp_ids = tgt_seq[j].tolist()
                # 移除开始标记
                hyp_ids = hyp_ids[1:]
                # 如果有结束标记，截断到结束标记
                if eos_id in hyp_ids:
                    hyp_ids = hyp_ids[:hyp_ids.index(eos_id)]
                # 移除padding
                hyp_ids = [id for id in hyp_ids if id != tgt_tokenizer.pad_token_id]
                hyp_tokens = tgt_tokenizer.convert_ids_to_tokens(hyp_ids)
                
                # 添加到BLEU评估列表
                references.append([ref_tokens])
                hypotheses.append(hyp_tokens)
    
    # 计算BLEU分数
    smoothing = SmoothingFunction().method1
    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smoothing)
    return bleu_score

def translate_sentence(model, src, src_tokenizer, tgt_tokenizer, device, max_length=100):
    """翻译单个句子"""
    model.eval()
    with torch.no_grad():
        # 编码源句子
        src_encoding = src_tokenizer(
            src,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
        src_tensor = src_encoding['input_ids'].to(device)
        
        # 生成序列的起始标记
        sos_id = tgt_tokenizer.cls_token_id
        eos_id = tgt_tokenizer.sep_token_id
        
        # 初始化目标序列为起始标记
        tgt_seq = torch.ones(1, 1).fill_(sos_id).long().to(device)
        
        # 自回归生成翻译
        for _ in range(max_length):
            output = model(src_tensor, tgt_seq)
            next_word = output[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt_seq = torch.cat([tgt_seq, next_word], dim=1)
            
            # 如果生成了结束标记，终止生成
            if next_word.item() == eos_id:
                break
        
        # 处理输出序列
        output_ids = tgt_seq.squeeze().tolist()[1:]  # 移除开始标记
        # 如果有结束标记，截断到结束标记处
        if eos_id in output_ids:
            output_ids = output_ids[:output_ids.index(eos_id)]
        
        # 解码获取最终文本
        translated_text = tgt_tokenizer.decode(output_ids, skip_special_tokens=True)
    
    return translated_text

class Translator:
    def __init__(self, model_path=None, model=None, src_tokenizer_name='bert-base-chinese', 
                 tgt_tokenizer_name='bert-base-uncased', max_length=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 使用与训练相同的预训练tokenizer
        self.src_tokenizer = BertTokenizer.from_pretrained(src_tokenizer_name)
        self.tgt_tokenizer = BertTokenizer.from_pretrained(tgt_tokenizer_name)
        self.max_length = max_length
        
        # 创建或加载模型
        if model is None:
            # 获取词汇量大小
            src_vocab_size = len(self.src_tokenizer)
            tgt_vocab_size = len(self.tgt_tokenizer)
            vocab_size = max(src_vocab_size, tgt_vocab_size)
            
            if model_path is not None:
                # 先加载检查点获取超参数
                state_dict, hyperparams = load_checkpoint(model_path, device=self.device)
                
                # 使用检查点中的超参数创建模型
                model_params = {
                    'd_model': hyperparams.get('D_MODEL', 256),  # 默认值设为训练时的值
                    'nhead': hyperparams.get('NHEAD', 4),
                    'num_encoder_layers': hyperparams.get('NUM_ENCODER_LAYERS', 6),
                    'num_decoder_layers': hyperparams.get('NUM_DECODER_LAYERS', 6),
                    'd_ff': hyperparams.get('D_FF', 512),
                    'vocab_size': hyperparams.get('VOCAB_SIZE', vocab_size),
                    'dropout': hyperparams.get('DROPOUT', 0.1),
                    'max_len': hyperparams.get('MAX_LEN', 512)
                }
                
                print("使用以下参数创建模型:")
                for k, v in model_params.items():
                    print(f"  {k}: {v}")
                    
                # 创建模型并加载权重
                self.model = Transformer(**model_params).to(self.device)
                self.model.load_state_dict(state_dict)
            else:
                # 没有提供模型路径，使用默认参数
                self.model = Transformer(
                    d_model=256,  # 使用训练中相同的值
                    nhead=4,
                    num_encoder_layers=6,
                    num_decoder_layers=6,
                    d_ff=512,
                    vocab_size=vocab_size,
                    dropout=0.1,
                    max_len=512
                ).to(self.device)
        else:
            # 直接使用提供的模型
            self.model = model
            
        self.model.eval()
    
    def _generate_square_subsequent_mask(self, sz):
        # build causal mask for decoder (1 = masked)
        mask = torch.triu(torch.ones(sz, sz, device=self.device), diagonal=1)
        return mask.bool()
        
    def predict(self, src_text, temperature=1.0, top_k=0, debug=True, greedy=False):
        """
        使用模型预测翻译结果
        
        Args:
            src_text: 源文本
            temperature: 温度参数，控制采样随机性
            top_k: 仅从概率最高的k个词中采样，0表示不使用此功能
            debug: 是否打印调试信息
            greedy: 是否使用贪婪解码（总是选择概率最高的token）
        """
        # 编码输入
        src_encoding = self.src_tokenizer(
            src_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        src = src_encoding['input_ids'].to(self.device)
        
        if debug:
            src_tokens = self.src_tokenizer.convert_ids_to_tokens(src[0])
            print(f"源文本分词: {src_tokens}")
            print(f"源文本ID: {src[0].tolist()}")
            
        # 使用自回归方式生成输出
        with torch.no_grad():
            # 解码器初始化 - 从[CLS]标记开始（对应BERT的起始标记）
            sos_id = self.tgt_tokenizer.cls_token_id
            eos_id = self.tgt_tokenizer.sep_token_id
            
            if debug:
                print(f"开始标记ID: {sos_id}, 结束标记ID: {eos_id}")
                # 打印词汇表大小
                print(f"目标词汇表大小: {len(self.tgt_tokenizer)}")
                
            tgt = torch.ones(1, 1).fill_(sos_id).long().to(self.device)
            
            generated_tokens = []
            # 自回归生成
            for i in range(self.max_length):
                # 使用模型的forward方法获取当前预测
                output = self.model(src, tgt)
                
                # 获取最后一个位置的预测
                logits = output[:, -1, :]
                
                if debug and i < 3:  # 只打印前几步的详细信息
                    # 打印概率分布的基本统计信息
                    probs = torch.softmax(logits, dim=-1)
                    top_probs, top_indices = torch.topk(probs, 5)
                    print(f"Step {i} - Top 5 tokens:")
                    for j, (idx, prob) in enumerate(zip(top_indices[0], top_probs[0])):
                        token = self.tgt_tokenizer.convert_ids_to_tokens([idx.item()])[0]
                        print(f"  {j+1}. Token: '{token}' (ID: {idx.item()}) - Prob: {prob.item():.4f}")
                
                # 应用温度
                if temperature != 1.0:
                    logits = logits / temperature
                
                # 应用top-k采样
                if top_k > 0:
                    top_k = min(top_k, logits.size(-1))  # 不能超过词汇表大小
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # 将logits转换为概率分布
                probs = torch.softmax(logits, dim=-1)
                
                if greedy:
                    # 贪婪解码 - 选择概率最高的token
                    next_word = torch.argmax(probs, dim=-1)
                else:
                    # 采样解码 - 根据概率分布随机采样
                    next_word = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
                # 将预测的token添加到目标序列
                tgt = torch.cat([tgt, next_word.unsqueeze(0)], dim=1)
                token_id = next_word.item()
                generated_tokens.append(token_id)
                
                if debug:
                    token = self.tgt_tokenizer.convert_ids_to_tokens([token_id])[0]
                    print(f"生成的token: '{token}' (ID: {token_id})")
                
                # 如果生成了结束标记，终止生成
                if token_id == eos_id:
                    break
        
        # 处理输出序列
        output_ids = tgt.squeeze().tolist()[1:]  # 移除开始标记
        # 如果序列中有结束标记，截断到结束标记处
        if eos_id in output_ids:
            output_ids = output_ids[:output_ids.index(eos_id)]
        
        # 解码获取最终文本
        return self.tgt_tokenizer.decode(output_ids, skip_special_tokens=True)

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
        "MAX_LENGTH": 10,
        "VOCAB_SIZE": vocab_size,
        
        # 模型参数
        "D_MODEL": 256,
        "NHEAD": 4,
        "NUM_ENCODER_LAYERS": 6,
        "NUM_DECODER_LAYERS": 6,
        "D_FF": 512,
        "DROPOUT": 0.1,
        "MAX_LEN": 10,
        
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
    
    # 修改数据集初始化，添加验证集和测试集
    train_dataset = TranslationDataset(
        src_path='data/little.src',  
        tgt_path='data/little.tgt',  
        src_tokenizer_name=src_tokenizer_name,
        tgt_tokenizer_name=tgt_tokenizer_name,
        max_length=HYPERPARAMETERS["MAX_LENGTH"]
    )
    train_loader = DataLoader(train_dataset, batch_size=HYPERPARAMETERS["BATCH_SIZE"], shuffle=True)
    
    val_dataset = TranslationDataset(
        src_path='data/val.src',  # 验证集中文文件路径
        tgt_path='data/val.tgt',  # 验证集英文文件路径
        src_tokenizer_name=src_tokenizer_name,
        tgt_tokenizer_name=tgt_tokenizer_name,
        max_length=HYPERPARAMETERS["MAX_LENGTH"]
    )
    val_loader = DataLoader(val_dataset, batch_size=HYPERPARAMETERS["BATCH_SIZE"], shuffle=False)
    
    test_dataset = TranslationDataset(
        src_path='data/test.src',  # 测试集中文文件路径
        tgt_path='data/test.tgt',  # 测试集英文文件路径
        src_tokenizer_name=src_tokenizer_name,
        tgt_tokenizer_name=tgt_tokenizer_name,
        max_length=HYPERPARAMETERS["MAX_LENGTH"]
    )
    test_loader = DataLoader(test_dataset, batch_size=HYPERPARAMETERS["BATCH_SIZE"], shuffle=False)
    
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
    
    # 修改训练循环，增加BLEU评估
    best_bleu = 0
    for epoch in range(start_epoch, HYPERPARAMETERS["EPOCHS"]):
        # 打印第一个训练样本的翻译结果
        print("\n" + "="*80)
        print(f"Epoch {epoch+1} - 示例翻译")
        print("="*80)
        
        # 获取第一个训练样本
        first_src_text = train_dataset.src_texts[0]
        first_tgt_text = train_dataset.tgt_texts[0]
        
        # 生成翻译
        model.eval()  # 临时设为评估模式
        translated_text = translate_sentence(model, first_src_text, src_tokenizer, tgt_tokenizer, device,max_length=HYPERPARAMETERS["MAX_LENGTH"])
        model.train()  # 恢复训练模式
        
        print(f"源文本 (中文): {first_src_text}")
        print(f"目标文本 (英文): {first_tgt_text}")
        print(f"模型翻译 (英文): {translated_text}")
        print("="*80 + "\n")
        
        # 开始正常的训练循环
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
        
        # 正常保存检查点
        if (epoch + 1) % CHECKPOINT_CONFIG["SAVE_EVERY"] == 0:
            checkpoint_path = os.path.join(CHECKPOINT_CONFIG["SAVE_DIR"], f'transformer_epoch{epoch+1}.pth')
            save_checkpoint(model, optimizer, lr_scheduler, epoch, HYPERPARAMETERS, checkpoint_path)
        
        print(f'Epoch [{epoch+1}/{HYPERPARAMETERS["EPOCHS"]}] Average Loss: {total_loss/len(train_loader):.4f}')
    
    # 训练完成后，对测试集进行评估
    print("\n训练完成！计算测试集BLEU分数...")
    test_bleu = calculate_bleu(model, test_loader, src_tokenizer, tgt_tokenizer, device)
    print(f'Final Test BLEU: {test_bleu:.4f}')
    
    # 加载最佳模型并进行测试集评估
    print("加载最佳模型进行最终评估...")
    best_model_path = os.path.join(CHECKPOINT_CONFIG["SAVE_DIR"], 'best_transformer_model.pth')
    if os.path.exists(best_model_path):
        model, _, _ = load_checkpoint(best_model_path, model)
        best_model_test_bleu = calculate_bleu(model, test_loader, src_tokenizer, tgt_tokenizer, device)
        print(f'Best Model Test BLEU: {best_model_test_bleu:.4f}')
    
    # 创建Translator实例用于交互式翻译
    translator = Translator(model=model, 
                          src_tokenizer_name=src_tokenizer_name,
                          tgt_tokenizer_name=tgt_tokenizer_name,
                          max_length=HYPERPARAMETERS["MAX_LENGTH"])
    
    # 进入交互式翻译模式
    print("\n进入交互式翻译模式。输入'q'退出。")
    while True:
        user_input = input("\n输入中文文本进行翻译: ")
        if user_input.lower() == 'q':
            break
        
        # 贪婪解码
        greedy_output = translator.predict(user_input, greedy=True, debug=False)
        print(f"贪婪解码翻译结果: {greedy_output}")
        
        # 采样解码
        sampling_output = translator.predict(user_input, temperature=0.8, top_k=5, greedy=False, debug=False)
        print(f"采样解码翻译结果: {sampling_output}")

if __name__ == '__main__':
    # 首先尝试训练模式
    if input("是否进入训练模式? (y/n): ").lower() == 'y':
        train()
    else:
        # 直接进入翻译模式
        model_path = input("请输入模型路径 (默认: transformer_epoch100.pth): ")
        if not model_path:
            model_path = "transformer_epoch100.pth"
            
        # 检查文件是否存在
        if not os.path.exists(model_path):
            print(f"错误: 模型文件 '{model_path}' 不存在。")
            exit(1)
            
        translator = Translator(
            model_path=model_path,
            src_tokenizer_name='bert-base-chinese',
            tgt_tokenizer_name='bert-base-uncased'
        )
        
        # 进入交互式翻译模式
        print("\n进入交互式翻译模式。输入'q'退出。")
        while True:
            user_input = input("\n输入中文文本进行翻译: ")
            if user_input.lower() == 'q':
                break
            
            # 贪婪解码
            greedy_output = translator.predict(user_input, greedy=True, debug=False)
            print(f"贪婪解码翻译结果: {greedy_output}")
            
            # 采样解码
            sampling_output = translator.predict(user_input, temperature=0.8, top_k=5, greedy=False, debug=False)
            print(f"采样解码翻译结果: {sampling_output}")
