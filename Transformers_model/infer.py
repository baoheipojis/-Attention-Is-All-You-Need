import torch
from tokenizers import Tokenizer, decoders
from model import Transformer
from tokenizer import build_bpe_tokenizer  # 假设已有分词器

class Translator:
    def __init__(self, model_path, tokenizer_path, max_length=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        # set decoder to handle byte-level decoding
        self.tokenizer.decoder = decoders.ByteLevel()
        self.max_length = max_length  # Add this line to store the parameter
        
        # 加载模型参数（需与训练参数一致）
        self.model = Transformer(
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            d_ff=2048,
            vocab_size=32000,
            dropout=0.1,
            max_len=512
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def _generate_square_subsequent_mask(self, sz):
        # build causal mask for decoder (1 = masked)
        mask = torch.triu(torch.ones(sz, sz, device=self.device), diagonal=1)
        return mask.bool()
        
    def predict(self, src_text):
        # 编码输入
        src_ids = self.tokenizer.encode(src_text).ids
        src = torch.LongTensor(src_ids).unsqueeze(0).to(self.device)
        
        # 添加位置编码
        src = self.model.input_embedding(src) * self.model.scale
        src = self.model.positional_encoding(src)
        
        # 修正编码器处理
        with torch.no_grad():
            # 处理编码器层
            for layer in self.model.encoder:
                src = layer(src, None)
            memory = src
            
            # 解码器初始化
            tgt = torch.ones(1,1).fill_(self.tokenizer.token_to_id("<sos>")).long().to(self.device)
            
            for _ in range(self.max_length):
                # 解码器处理
                tgt_embed = self.model.output_embedding(tgt) * self.model.scale
                tgt_embed = self.model.positional_encoding(tgt_embed)
                
                # generate casual mask for current tgt length
                tgt_mask = self._generate_square_subsequent_mask(tgt.size(1))
                
                for layer in self.model.decoder:
                    tgt_embed = layer(tgt_embed, memory, None, tgt_mask)
                
                output = self.model.fc_out(tgt_embed)
                prob = output[:, -1, :]
                next_word = torch.argmax(prob, dim=-1)
                tgt = torch.cat([tgt, next_word.unsqueeze(0)], dim=1)
                
                if next_word == self.tokenizer.token_to_id("<eos>"):
                    break
            
        # strip off the initial <sos> before decoding
        output_ids = tgt.squeeze().tolist()
        if output_ids and output_ids[0] == self.tokenizer.token_to_id("<sos>"):
            output_ids = output_ids[1:]
        # decode and remove special tokens to get clean text
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

if __name__ == "__main__":
    translator = Translator(
        model_path="transformer_epoch5.pth",
        tokenizer_path="bpe_tokenizer.json"
    )
    
    # 示例用法
    test_sentence = "你好，世界"
    print(f"Input: {test_sentence}")
    print(f"Output: {translator.predict(test_sentence)}")