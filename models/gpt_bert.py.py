import torch  
import torch.nn as nn  
import torch.nn.functional as F
import jieba
from typing import List, Dict, Optional

class TextProcessor:
    def __init__(self, vocab_file: str = None, max_len: int = 512):
        self.vocab_file = vocab_file
        self.max_len = max_len
        # 示例新闻文本数据
        self.example_texts = [
            "国务院召开常务会议，部署促进经济稳定增长的措施",
            "科技创新助力产业升级，人工智能应用广泛普及",
            "2024年全球经济展望：机遇与挑战并存",
            "环保政策推动绿色发展，可再生能源产业蓬勃发展"
        ]
    
    def tokenize(self, text: str) -> List[str]:
        """使用jieba分词处理文本"""
        return list(jieba.cut(text))
    
    def get_example_batch(self, batch_size: int = 4) -> Dict[str, torch.Tensor]:
        """获取示例批次数据"""
        input_ids = []
        attention_mask = []
        
        for text in self.example_texts[:batch_size]:
            tokens = self.tokenize(text)
            # 简单的数字化处理(实际应用中需要使用词表)
            ids = [i+1 for i in range(len(tokens))]  # 0作为padding
            mask = [1] * len(ids)
            
            # padding处理
            if len(ids) < self.max_len:
                ids.extend([0] * (self.max_len - len(ids)))
                mask.extend([0] * (self.max_len - len(mask)))
            else:
                ids = ids[:self.max_len]
                mask = mask[:self.max_len]
                
            input_ids.append(ids)
            attention_mask.append(mask)
            
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask)
        }

class PositionalEncoding(nn.Module):  
    def __init__(self, d_model: int, max_len: int = 5000):  
        super(PositionalEncoding, self).__init__()  
        pe = torch.zeros(max_len, d_model)  
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))  
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)  
        pe = pe.unsqueeze(0).transpose(0, 1)  
        self.register_buffer('pe', pe)  

    def forward(self, x):  
        x = x + self.pe[:x.size(0), :]  
        return x  

class TransformerEncoderLayer(nn.Module):  
    def __init__(self, d_model: int, nhead: int):  
        super(TransformerEncoderLayer, self).__init__()  
        self.self_attn = nn.MultiheadAttention(d_model, nhead)  
        self.ffn = nn.Sequential(  
            nn.Linear(d_model, d_model * 4),  
            nn.ReLU(),  
            nn.Linear(d_model * 4, d_model)  
        )  
        self.norm1 = nn.LayerNorm(d_model)  
        self.norm2 = nn.LayerNorm(d_model)  
        self.dropout = nn.Dropout(0.1)  

    def forward(self, src, src_mask=None):  
        x = src  
        x2 = self.self_attn(x, x, x, attn_mask=src_mask)[0]  
        x = x + self.dropout(x2)  
        x = self.norm1(x)  
        x2 = self.ffn(x)  
        x = x + self.dropout(x2)  
        x = self.norm2(x)  
        return x  

class TransformerEncoder(nn.Module):  
    def __init__(self, d_model: int, nhead: int, num_layers: int):  
        super(TransformerEncoder, self).__init__()  
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, nhead) for _ in range(num_layers)])  
        self.positional_encoding = PositionalEncoding(d_model)  

    def forward(self, src):  
        src = self.positional_encoding(src)  
        for layer in self.layers:  
            src = layer(src)  
        return src  

class MaskedLanguageModel(nn.Module):  
    def __init__(self, vocab_size: int, d_model: int):  
        super(MaskedLanguageModel, self).__init__()  
        self.embedding = nn.Embedding(vocab_size, d_model)  
        self.transformer_encoder = TransformerEncoder(d_model, nhead=8, num_layers=6)  
        self.fc = nn.Linear(d_model, vocab_size)  

    def forward(self, input_ids):  
        # Generate mask for transformer  
        src_mask = (input_ids == 0).unsqueeze(1).unsqueeze(2)  # 0 represents padding  
        x = self.embedding(input_ids)  
        x = self.transformer_encoder(x)  
        x = self.fc(x)  
        return x  

# 创建文本处理器和模型实例
text_processor = TextProcessor()
vocab_size = 30000  # 词汇表大小  
d_model = 256  # 模型维度  

# 创建模型实例  
model = MaskedLanguageModel(vocab_size, d_model)  

# 准备真实的中文文本数据
batch = text_processor.get_example_batch(batch_size=2)
input_ids = batch['input_ids']
attention_mask = batch['attention_mask']

# 前向传播  
outputs = model(input_ids)  
predicted_token = outputs.argmax(dim=-1)  

print("输入的新闻文本:", text_processor.example_texts[:2])
print("模型输出维度:", outputs.shape)  


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(GPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_encoder = TransformerEncoder(d_model, nhead, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.positional_encoding(x)
        print("before transformer encoder",x.shape)
        x = self.transformer_encoder(x)
        print("after transformer encoder",x.shape)
        print(self.fc(x).shape)
        x = self.fc(x)
        return x
# 参数设置
vocab_size = 30000  # 词汇表大小
d_model = 256  # 模型维度
seq_length = 20  # 序列长度
nhead = 8  # 多头注意力的头数
num_layers = 6  # Transformer编码器的层数
# 创建模型实例
model = GPT(vocab_size, d_model, nhead, num_layers)
# 准备数据
input_ids = torch.randint(1, vocab_size, (1, seq_length)).long()  # 随机生成输入序列
# 前向传播
outputs = model(input_ids)
predicted_token = outputs.argmax(dim=-1)
print("Input IDs:", input_ids)
print("Predicted Tokens:", predicted_token)
