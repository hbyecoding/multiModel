import torch
import logging
from transformers import BertTokenizer, BertModel
from typing import List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self):
        """初始化文本处理器"""
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
    def process(self, texts: List[str]) -> torch.Tensor:
        """处理文本列表并提取特征
        
        Args:
            texts: 文本列表
            
        Returns:
            features: 文本特征张量
        """
        features = []
        
        with torch.no_grad():
            for text in texts:
                # 编码文本
                inputs = self.tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                # 获取BERT特征
                outputs = self.model(**inputs)
                
                # 使用[CLS]标记的输出作为文本特征
                feature = outputs.last_hidden_state[:, 0, :]
                features.append(feature)
        
        # 堆叠所有特征
        features = torch.cat(features, dim=0)
        return features
    
    def batch_process(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """批量处理文本
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
            
        Returns:
            features: 文本特征张量
        """
        features = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_features = self.process(batch_texts)
            features.append(batch_features)
        
        features = torch.cat(features, dim=0)
        return features
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的相似度
        
        Args:
            text1: 第一段文本
            text2: 第二段文本
            
        Returns:
            相似度分数
        """
        features = self.process([text1, text2])
        similarity = torch.cosine_similarity(features[0], features[1], dim=0)
        return similarity.item()
    
    def analyze_title(self, title: str) -> Dict[str, Any]:
        """分析视频标题
        
        Args:
            title: 视频标题
            
        Returns:
            标题分析结果
        """
        # 提取标题特征
        features = self.process([title])[0]
        
        # 计算特征统计量
        stats = {
            'feature_mean': features.mean().item(),
            'feature_std': features.std().item(),
            'feature_norm': torch.norm(features).item()
        }
        
        return {
            'features': features.numpy(),
            'stats': stats,
            'length': len(title)
        }