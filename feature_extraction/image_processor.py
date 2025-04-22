import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from typing import Dict, Any
import clip
import numpy as np

class ImageProcessor:
    def __init__(self, clip_model: str = 'ViT-B/32'):
        """初始化图像处理器
        
        Args:
            clip_model: CLIP模型名称
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.preprocess = clip.load(clip_model, device=self.device)
        
    def extract_features(self, image_path: str) -> torch.Tensor:
        """提取图像特征
        
        Args:
            image_path: 图像路径
            
        Returns:
            图像特征张量
        """
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            
        return image_features
    
    def compute_image_text_similarity(self, image_path: str, text: str) -> float:
        """计算图像和文本的相似度
        
        Args:
            image_path: 图像路径
            text: 文本内容
            
        Returns:
            相似度分数
        """
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        text_token = clip.tokenize([text]).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_token)
            
            similarity = torch.cosine_similarity(image_features, text_features)
            
        return similarity.item()
    
    def analyze_thumbnail(self, image_path: str) -> Dict[str, Any]:
        """分析视频封面
        
        Args:
            image_path: 封面图像路径
            
        Returns:
            封面分析结果
        """
        # 提取图像特征
        features = self.extract_features(image_path)
        
        # 计算特征统计量
        stats = {
            'feature_mean': features.mean().item(),
            'feature_std': features.std().item(),
            'feature_norm': torch.norm(features).item()
        }
        
        # 加载原始图像获取基本信息
        image = Image.open(image_path)
        
        return {
            'features': features.cpu().numpy(),
            'stats': stats,
            'size': image.size,
            'mode': image.mode
        }