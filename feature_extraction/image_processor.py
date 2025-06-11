import torch
import logging
from PIL import Image
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTModel
from typing import List, Union, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        """初始化图像处理器"""
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # 设置图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """加载图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            image: PIL图像对象
        """
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            logger.error(f"加载图像时出错 {image_path}: {str(e)}")
            return None
    
    def process_single_image(self, image: Image.Image) -> torch.Tensor:
        """处理单张图像
        
        Args:
            image: PIL图像对象
            
        Returns:
            features: 图像特征张量
        """
        if image is None:
            return None
        
        # 预处理图像
        inputs = self.feature_extractor(
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]  # 使用[CLS]标记
        
        return features
    
    def process(self, image_paths: List[Union[str, Path]]) -> torch.Tensor:
        """处理图像列表
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            features: 图像特征张量
        """
        features = []
        
        for path in image_paths:
            # 加载图像
            image = self.load_image(path)
            if image is None:
                continue
                
            # 处理图像
            feature = self.process_single_image(image)
            if feature is not None:
                features.append(feature)
        
        if not features:
            raise ValueError("没有成功处理任何图像")
        
        # 堆叠所有特征
        features = torch.cat(features, dim=0)
        return features
    
    def batch_process(self, image_paths: List[Union[str, Path]], 
                     batch_size: int = 32) -> torch.Tensor:
        """批量处理图像
        
        Args:
            image_paths: 图像路径列表
            batch_size: 批次大小
            
        Returns:
            features: 图像特征张量
        """
        features = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_features = self.process(batch_paths)
            features.append(batch_features)
        
        features = torch.cat(features, dim=0)
        return features
    
    def compute_similarity(self, image1_path: Union[str, Path],
                         image2_path: Union[str, Path]) -> float:
        """计算两张图像的相似度
        
        Args:
            image1_path: 第一张图像路径
            image2_path: 第二张图像路径
            
        Returns:
            similarity: 相似度分数
        """
        # 加载并处理图像
        image1 = self.load_image(image1_path)
        image2 = self.load_image(image2_path)
        
        if image1 is None or image2 is None:
            return 0.0
        
        # 提取特征
        feature1 = self.process_single_image(image1)
        feature2 = self.process_single_image(image2)
        
        # 计算余弦相似度
        similarity = torch.cosine_similarity(feature1, feature2, dim=1)
        return similarity.item()