import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any
import numpy as np

class TextProcessor:
    def __init__(self, model_name: str = 'bert-base-chinese'):
        """初始化文本处理器
        
        Args:
            model_name: 预训练模型名称
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
    def extract_features(self, texts: List[str]) -> torch.Tensor:
        """提取文本特征
        
        Args:
            texts: 文本列表
            
        Returns:
            文本特征张量
        """
        inputs = self.tokenizer(texts, padding=True, truncation=True, 
                              max_length=512, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用[CLS]标记的输出作为文本特征
            features = outputs.last_hidden_state[:, 0, :]
            
        return features
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的相似度
        
        Args:
            text1: 第一段文本
            text2: 第二段文本
            
        Returns:
            相似度分数
        """
        features = self.extract_features([text1, text2])
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
        features = self.extract_features([title])[0]
        
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