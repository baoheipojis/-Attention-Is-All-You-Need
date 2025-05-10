import torch
from transformers import BertTokenizer
from model import Transformer

class Translator:
    def __init__(self, model_path, src_tokenizer_name='bert-base-chinese', 
                 tgt_tokenizer_name='bert-base-uncased', max_length=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 使用与训练相同的预训练tokenizer
        self.src_tokenizer = BertTokenizer.from_pretrained(src_tokenizer_name)
        self.tgt_tokenizer = BertTokenizer.from_pretrained(tgt_tokenizer_name)
        self.max_length = max_length
        
        # 获取词汇量大小
        src_vocab_size = len(self.src_tokenizer)
        tgt_vocab_size = len(self.tgt_tokenizer)
        vocab_size = max(src_vocab_size, tgt_vocab_size)
        
        # 加载模型参数（需与训练参数一致）
        self.model = Transformer(
            d_model=128,
            nhead=4,
            num_encoder_layers=6,
            num_decoder_layers=6,
            d_ff=512,
            vocab_size=vocab_size,
            dropout=0.1,
            max_len=512
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))  # 添加 weights_only=True 以解决警告
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

if __name__ == "__main__":
    translator = Translator(
        model_path="transformer_epoch5.pth",
        src_tokenizer_name='bert-base-chinese',
        tgt_tokenizer_name='bert-base-uncased'
    )
    
    # 示例用法
    test_sentence = "你好，世界"
    print(f"Input: {test_sentence}")
    
    # 仅使用贪婪解码
    print("\n--- 贪婪解码 ---")
    output = translator.predict(test_sentence, greedy=True)
    print(f"Output: {output}")