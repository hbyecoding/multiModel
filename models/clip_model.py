import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from typing import Dict, Any, Union
import numpy as np

class CLIPModel:
    def __init__(self):
        """初始化CLIP模型"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
        
        # 简化版的图像编码器
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512)
        ).to(self.device)
        
        # 简化版的文本编码器
        self.text_encoder = nn.Sequential(
            nn.Embedding(30000, 256),  # 假设词汇表大小为30000
            nn.LSTM(256, 512, batch_first=True),
            nn.Linear(512, 512)
        ).to(self.device)
        
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """预处理图像
        
        Args:
            image: 图像路径、numpy数组或PIL图像
            
        Returns:
            预处理后的图像张量
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
        return image_tensor
    
    def encode_image(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """编码图像
        
        Args:
            image: 图像路径、numpy数组或PIL图像
            
        Returns:
            图像特征向量
        """
        image_tensor = self.preprocess_image(image)
        with torch.no_grad():
            image_features = self.image_encoder(image_tensor)
        return image_features
    
    def encode_text(self, text: str) -> torch.Tensor:
        """编码文本
        
        Args:
            text: 输入文本
            
        Returns:
            文本特征向量
        """
        # 简单的文本处理，实际应用中需要更复杂的分词和编码
        tokens = torch.randint(0, 30000, (1, 50)).to(self.device)  # 模拟分词结果
        
        with torch.no_grad():
            text_features, _ = self.text_encoder(tokens)
            text_features = text_features.mean(dim=1)  # 平均池化
        return text_features
    
    def compute_similarity(self, text: str, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """计算文本和图像的相似度
        
        Args:
            text: 输入文本
            image: 图像路径、numpy数组或PIL图像
            
        Returns:
            相似度分数和匹配结果
        """
        # 获取特征向量
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        # 归一化特征
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 计算余弦相似度
        similarity = torch.matmul(image_features, text_features.t()).item()
        
        return {
            'similarity_score': similarity,
            'is_match': similarity > 0.5,  # 简单的匹配阈值
            'confidence': abs(similarity)
        }